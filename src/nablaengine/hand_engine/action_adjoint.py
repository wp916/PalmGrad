"""单步动作VJP：原积分反传、Q/H隐式线性解；不是状态/多步完整反向。

固定s_t、PalmMotion及kp/kv；对command.values求导。接触/限位/干摩擦
采用前向末次row_piece分支。求导目标为收敛最优解，而非Newton迭代程序。
"""
import ctypes
from dataclasses import dataclass
from pathlib import Path
import tempfile
from types import SimpleNamespace
import numpy as np
from nablaengine.cuda import compile_cuda
from nablaengine.g1_engine import source
from .integration_adjoint import CudaIntegrationAdjoint, _original


CUDA=r'''
__global__ void integrate_pullback(float h,const float* input,const int* meta,float* output){
 const float* q=input;const float* v=q+NQ;const float* adv=v+NV;
 const float* qb=adv+NV;const float* vb=qb+NQ;
 float grad[NV];for(int i=0;i<NV;i++)grad[i]=vb[i];
 for(int b=1;b<NB;b++){
   int kind=meta[3*NB+b],qi=meta[NB+b],vi=meta[2*NB+b];
   if(kind==2)grad[vi]+=h*qb[qi];
   else if(kind==3){
     for(int k=0;k<3;k++)grad[vi+k]+=h*qb[qi+k];
     float w[3],qbar[4],wbar[3];for(int k=0;k<3;k++)w[k]=fmaf(h,adv[vi+3+k],v[vi+3+k]);
     quat_integrate_vjp(q+qi+3,w,h,qb+qi+3,qbar,wbar);
     for(int k=0;k<3;k++)grad[vi+3+k]+=wbar[k];
   }
 }
 for(int i=0;i<NV;i++)output[i]=h*grad[i];
}
__global__ void filter_pullback(float h,const float* input,float* output){
 const float* abar=input;const float* coeff=abar+NV;const float* y=coeff+NV;
 for(int i=threadIdx.x;i<NV;i+=32)output[i]=abar[i]-h*coeff[i]*y[i];
}
extern "C" int run(int mode,float h,const float* input,const int* meta,float* output){
 float *di=nullptr,*dout=nullptr;int* dm=nullptr;cudaError_t e;
 #define ACHECK(call) if((e=(call))!=cudaSuccess){cudaFree(di);cudaFree(dout);cudaFree(dm);return(int)e;}
 int n=mode==0?2*NQ+3*NV:3*NV;
 ACHECK(cudaMalloc(&di,n*sizeof(float)));ACHECK(cudaMalloc(&dout,NV*sizeof(float)));
 ACHECK(cudaMalloc(&dm,4*NB*sizeof(int)));
 ACHECK(cudaMemcpy(di,input,n*sizeof(float),cudaMemcpyHostToDevice));
 ACHECK(cudaMemcpy(dm,meta,4*NB*sizeof(int),cudaMemcpyHostToDevice));
 if(mode==0)integrate_pullback<<<1,1>>>(h,di,dm,dout);else filter_pullback<<<1,32>>>(h,di,dout);
 ACHECK(cudaGetLastError());ACHECK(cudaDeviceSynchronize());
 ACHECK(cudaMemcpy(output,dout,NV*sizeof(float),cudaMemcpyDeviceToHost));
 cudaFree(di);cudaFree(dout);cudaFree(dm);return 0;
}
'''


@dataclass(frozen=True)
class ActionTape:
    """由record生成；不接受任意前向结果与不同输入拼接。"""
    inputs: object
    forward: object
    implicit_matrix: np.ndarray
    _owner: object


@dataclass(frozen=True)
class ActionVJP:
    values: np.ndarray
    advancement_cotangent: np.ndarray
    acceleration_cotangent: np.ndarray
    stationarity_scaled: float
    Q_relative_residual: float
    H_relative_residual: float
    convention: str = 'converged_solution_fixed_branch_action_vjp'


class CudaActionAdjoint:
    """record -> tape; vjp(tape,qpos_bar,qvel_bar) -> dL/dcommand.values。

    原G1 quat_integrate_vjp原文复用；Q/H解复用原CUDA Cholesky。
    不对warm_start、状态、手掌、材料或PD增益求导，不自动注册JAX/PyTorch。
    """
    def __init__(self,engine):
        self.engine=engine;self.numeric=engine.numeric;self.layout=engine.layout;self._owner=object()
        self.integration=CudaIntegrationAdjoint(engine.dynamics,numerics=engine.numeric.mode)
        m=SimpleNamespace(nbody=engine.dynamics.motion.nb,nq=self.layout.nq,nv=self.layout.nv)
        helpers='\n'.join(_original(n) for n in ('dot4','adj_sinc','adj_dsinc_x','quat_integrate_vjp'))
        code=source.header(m,1,1)+source.DEVICE+helpers+CUDA
        with tempfile.TemporaryDirectory(prefix='nabla_hand_action_adjoint_') as tmp:
            cu,so=Path(tmp)/'adjoint.cu',Path(tmp)/'adjoint.so';cu.write_text(self.numeric.cuda(code))
            compile_cuda(cu,so,extra=('-shared','-Xcompiler','-fPIC'));self._library=ctypes.CDLL(str(so))
        self._run=self._library.run;fp=self.numeric.pointer;ip=ctypes.POINTER(ctypes.c_int)
        self._run.argtypes=[ctypes.c_int,self.numeric.scalar,fp,ip,fp];self._run.restype=ctypes.c_int

    def record(self,inputs,*,warm_start=None,time=0.):
        with self.engine._lock:
            result=self.engine.step(inputs,warm_start=warm_start,time=time)
            # 用同一CUDA装配核重建Q，避免CPU乘加舍入与实际前向矩阵不一致。
            n=self.layout.nv;zero=np.zeros(n);kv=zero.copy()
            if inputs.command.mode=='position_pd':kv[:self.layout.hand_dofs]=inputs.command.kv
            _,Q,_=self.engine._prepare(result.mass_matrix,inputs.state,zero,zero,kv,zero)
            Q.setflags(write=False)
            return ActionTape(inputs,result,Q,self._owner)

    def _kernel(self,mode,values):
        packed=self.numeric.finite(values,'动作反传输入');out=np.empty(self.layout.nv,self.numeric.dtype)
        fp=self.numeric.pointer;ip=ctypes.POINTER(ctypes.c_int)
        status=self._run(mode,self.engine.h,packed.ctypes.data_as(fp),
                         self.engine.dynamics.motion.meta.ctypes.data_as(ip),out.ctypes.data_as(fp))
        if status or not np.isfinite(out).all():raise RuntimeError(f'CUDA动作反传失败:{status}')
        return out

    def vjp(self,tape,qpos_bar,qvel_bar):
        if not isinstance(tape,ActionTape) or tape._owner is not self._owner:raise ValueError('tape不属于此反向实例')
        with self.engine._lock:
            if self.engine.closed:raise RuntimeError('不能使用已关闭的engine反传')
            qb=self.numeric.finite(qpos_bar,'qpos_bar');vb=self.numeric.finite(qvel_bar,'qvel_bar')
            if qb.shape!=(self.layout.nq,) or vb.shape!=(self.layout.nv,):raise ValueError('状态余切形状错误')
            inputs,r=tape.inputs,tape.forward;s=inputs.state;n=self.layout.nv;h=self.engine.h
            abar=self.integration.evaluate(s,r.advancement_acceleration,h,qb,vb).qacc
            Q=tape.implicit_matrix
            y=self.engine.solver.linear_solve(Q,abar)
            astar_bar=self._kernel(1,np.r_[abar,r.velocity_coefficient,y])
            H=r.newton.evaluation.hessian
            lam=self.engine.solver.linear_solve(H,astar_bar)
            gain=(inputs.command.kp if inputs.command.mode=='position_pd' else np.ones(self.layout.hand_dofs))
            values=self.numeric.finite(lam[:self.layout.hand_dofs]*gain,'动作梯度')
            def residual(A,x,b):
                A=A.astype(float);x=x.astype(float);b=b.astype(float)
                return float(np.max(abs(A@x-b))/(1+np.max(abs(b))+np.max(np.sum(abs(A),axis=1))*np.max(abs(x))))
            qr,hr=residual(Q,y,abar),residual(H,lam,astar_bar)
            if max(qr,hr)>3e-5:raise RuntimeError('动作反传线性解残差过大')
            for a in (values,abar,astar_bar):a.setflags(write=False)
            return ActionVJP(values,abar,astar_bar,r.newton.scaled_gradient,qr,hr)

    def jacobian(self,tape):
        """显式d[qpos_next,qvel_next]/dvalues，按输出基向量反传；正确性入口。

        行数nq+nv，列数hand_dofs。会执行nq+nv次VJP，训练通常应直接用vjp。
        """
        nq,nv=self.layout.nq,self.layout.nv
        rows=[]
        for i in range(nq+nv):
            basis=np.zeros(nq+nv);basis[i]=1
            rows.append(self.vjp(tape,basis[:nq],basis[nq:]).values)
        result=np.array(rows);result.setflags(write=False);return result
