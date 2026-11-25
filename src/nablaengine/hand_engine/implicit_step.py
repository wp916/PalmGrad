"""外部几何驱动的hand implicitfast单步；主机调度的CUDA正确性入口。

实际CUDA计算RNE/M、约束行/Newton、(M+hKv)修正及原根积分。
不调用MuJoCo动力学/求解器；碰撞检测由显式provider提供，尚非完整批量生产后端。
"""
import ctypes
from dataclasses import dataclass
from pathlib import Path
import tempfile
import threading
import numpy as np
from nablaengine.cuda import compile_cuda
from nablaengine.model.hand import HandInput
from .backend import make_backend_plan
from .dynamics import CudaDynamics
from .constraint_table import CudaConstraintTable
from .newton import CudaNewton
from .time_step import CudaStateIntegrator
from .numerics import Numerics


CUDA=r'''
#include <cuda_runtime.h>
#define INPUT (NV*NV+6*NV)
#define OUTPUT (NV*NV+2*NV)
__global__ void prepare(float h,const float* input,float* output){
  const float* M=input;const float* v=M+NV*NV;const float* bias=v+NV;
  const float* effort=bias+NV;const float* damping=effort+NV;const float* kv=damping+NV;const float* astar=kv+NV;
  float* rhs=output;float* implicit=rhs+NV;float* advanced_rhs=implicit+NV*NV;
  for(int i=threadIdx.x;i<NV*NV;i+=32){
    implicit[i]=M[i];if(i/NV==i%NV)implicit[i]+=h*(damping[i/NV]+kv[i/NV]);
  }
  for(int i=threadIdx.x;i<NV;i+=32){
    rhs[i]=effort[i]-bias[i]-damping[i]*v[i];
    float value=0.f;for(int j=0;j<NV;j++)value+=M[i*NV+j]*astar[j];advanced_rhs[i]=value;
  }
}
extern "C" int run(float h,const float* input,float* output){
  float *di=nullptr,*dout=nullptr;cudaError_t e;
  #define ICHECK(call) if((e=(call))!=cudaSuccess){cudaFree(di);cudaFree(dout);return(int)e;}
  ICHECK(cudaMalloc(&di,INPUT*sizeof(float)));ICHECK(cudaMalloc(&dout,OUTPUT*sizeof(float)));
  ICHECK(cudaMemcpy(di,input,INPUT*sizeof(float),cudaMemcpyHostToDevice));
  prepare<<<1,32>>>(h,di,dout);ICHECK(cudaGetLastError());ICHECK(cudaDeviceSynchronize());
  ICHECK(cudaMemcpy(output,dout,OUTPUT*sizeof(float),cudaMemcpyDeviceToHost));
  cudaFree(di);cudaFree(dout);return 0;
}
'''

WORKSPACE=r'''
struct ImplicitWorkspace {float *input=nullptr,*output=nullptr;};
extern "C" int destroy_workspace(void* handle){
 if(!handle)return 0;ImplicitWorkspace* w=(ImplicitWorkspace*)handle;
 cudaError_t a=cudaFree(w->input),b=cudaFree(w->output);delete w;
 return(int)(a!=cudaSuccess?a:b);
}
extern "C" int create_workspace(void** handle){
 *handle=nullptr;ImplicitWorkspace* w=new ImplicitWorkspace;cudaError_t e;
 e=cudaMalloc(&w->input,INPUT*sizeof(float));
 if(e==cudaSuccess)e=cudaMalloc(&w->output,OUTPUT*sizeof(float));
 if(e!=cudaSuccess){destroy_workspace(w);return(int)e;}*handle=w;return 0;
}
extern "C" int run_workspace(void* handle,float h,const float* input,float* output){
 if(!handle)return(int)cudaErrorInvalidValue;
 ImplicitWorkspace* w=(ImplicitWorkspace*)handle;cudaError_t e;
 #define PCHECK(call) if((e=(call))!=cudaSuccess)return(int)e;
 PCHECK(cudaMemcpy(w->input,input,INPUT*sizeof(float),cudaMemcpyHostToDevice));
 prepare<<<1,32>>>(h,w->input,w->output);PCHECK(cudaGetLastError());PCHECK(cudaDeviceSynchronize());
 PCHECK(cudaMemcpy(output,w->output,OUTPUT*sizeof(float),cudaMemcpyDeviceToHost));return 0;
}
'''


@dataclass(frozen=True)
class ImplicitStepResult:
    """state/time位于步末，其余动力学与约束量位于步初。"""
    state: object
    time: float
    warm_start: np.ndarray
    constrained_acceleration: np.ndarray
    advancement_acceleration: np.ndarray
    free_acceleration: np.ndarray
    mass_matrix: np.ndarray
    velocity_coefficient: np.ndarray
    constraints: object
    newton: object
    implicit_residual_inf: float
    integration_convention: str = 'nabla_implicitfast'


class CudaImplicitFastStepper:
    """同一时刻HandInput -> 步末state；h取模型，手掌不在内部积分。

    provider(state,palm)必须返回当前几何的(ContactFrame序列,pair_ids)，每步重新调用。
    可传入collision.CudaPrimitiveContacts，使用已支持基本形状的GPU碰撞。
    也可显式使用step_with_contacts；本类不缓存或猜测过期接触。
    command为无饱和torque或PD，粘性damping来自模型；非零刚度、重力补偿、流体、限幅等拒绝。
    warm_start保存a_star，不是a_adv；两者分别返回，不承诺Newton满足严格梯度容差。
    本入口只接通已声明范围，不表示所有MuJoCo物理选项已完成能力审计。
    积分约定保留原Nabla，不含MuJoCo 3.17对独立free刚体额外加入的局部偏置导数LU修正。
    """
    def __init__(self,model,layout,contact_provider=None,*,capacity=None,numerics="float32"):
        self.numeric=Numerics(numerics)
        self._lock=threading.RLock();self._closed=False;self._handle=ctypes.c_void_p()
        if contact_provider is not None and not callable(contact_provider):raise TypeError('contact_provider必须可调用')
        if model.opt.noslip_iterations:raise ValueError('当前不支持noslip后处理')
        if np.any(model.jnt_stiffness) or np.any(model.body_gravcomp):raise ValueError('当前不支持非零关节弹簧或重力补偿')
        if model.opt.density or model.opt.viscosity or np.any(model.opt.wind):raise ValueError('当前不支持流体/风力')
        if int(model.opt.disableflags) or int(model.opt.enableflags):raise ValueError('当前不支持非默认enable/disableflags')
        if np.any(model.actuator_forcelimited) or np.any(model.actuator_ctrllimited) or np.any(model.jnt_actfrclimited):
            raise ValueError('当前HandInput是无饱和命令，不支持控制或力矩限幅')
        self.plan=make_backend_plan(model,layout,capacity);self.layout=layout;self.h=self.plan.physics['timestep']
        self.contact_provider=contact_provider;self.dynamics=CudaDynamics(model,layout,numerics=numerics)
        self.tables=CudaConstraintTable(model,self.plan,self.dynamics.motion);self.solver=CudaNewton(self.plan,numerics=numerics)
        self.integrator=CudaStateIntegrator(self.dynamics,numerics=numerics)
        damping=np.empty(layout.nv)
        for b in layout.blocks:damping[b.qvel_slice]=model.dof_damping[b.source_qvel_start:b.source_qvel_start+b.nv]
        self.damping=self.numeric.finite(damping,'粘性阻尼')
        if np.any(self.damping<0):raise ValueError('粘性阻尼必须非负')
        self.damping.setflags(write=False)
        with tempfile.TemporaryDirectory(prefix='nabla_hand_implicit_') as tmp:
            cu,so=Path(tmp)/'implicit.cu',Path(tmp)/'implicit.so';cu.write_text(self.numeric.cuda(f'#define NV {layout.nv}\n'+CUDA+WORKSPACE))
            compile_cuda(cu,so,extra=('-shared','-Xcompiler','-fPIC'));self._library=ctypes.CDLL(str(so))
        self._run=self._library.run;fp=self.numeric.pointer
        self._run.argtypes=[self.numeric.scalar,fp,fp];self._run.restype=ctypes.c_int
        self._create=self._library.create_workspace
        self._create.argtypes=[ctypes.POINTER(ctypes.c_void_p)];self._create.restype=ctypes.c_int
        self._execute=self._library.run_workspace
        self._execute.argtypes=[ctypes.c_void_p,self.numeric.scalar,fp,fp];self._execute.restype=ctypes.c_int
        self._destroy=self._library.destroy_workspace
        self._destroy.argtypes=[ctypes.c_void_p];self._destroy.restype=ctypes.c_int
        status=self._create(ctypes.byref(self._handle))
        if status:raise RuntimeError(f'CUDA implicitfast工作区创建失败:{status}')

    @property
    def closed(self):
        return self._closed

    @property
    def workspace_bytes(self):
        """只统计implicit装配的输入/输出显存，不代表整个engine显存。"""
        n=self.layout.nv
        return 0 if self._closed else np.dtype(self.numeric.dtype).itemsize*(2*n*n+8*n)

    def close(self):
        """释放自身装配工作区；外部传入的contact_provider仍由调用方管理。"""
        with self._lock:
            if self._closed:return
            self._closed=True
            if self._handle.value:
                handle=self._handle;self._handle=ctypes.c_void_p()
                status=self._destroy(handle)
                if status:raise RuntimeError(f'CUDA implicitfast工作区释放失败:{status}')

    def __enter__(self):
        if self._closed:raise RuntimeError('CUDA implicitfast工作区已关闭')
        return self

    def __exit__(self,*args):self.close()

    def __del__(self):
        try:self.close()
        except Exception:pass

    def _prepare(self,M,state,bias,effort,kv,astar):
        with self._lock:
            if self._closed:raise RuntimeError('CUDA implicitfast工作区已关闭')
            return self._prepare_locked(M,state,bias,effort,kv,astar)

    def _prepare_locked(self,M,state,bias,effort,kv,astar):
        nv=self.layout.nv
        packed=self.numeric.finite(np.r_[M.ravel(),state.qvel,bias,effort,self.damping,kv,astar],'implicitfast输入')
        out=np.empty(nv*nv+2*nv,self.numeric.dtype);fp=self.numeric.pointer
        status=self._execute(self._handle,self.h,packed.ctypes.data_as(fp),out.ctypes.data_as(fp))
        if status or not np.isfinite(out).all():raise RuntimeError(f'CUDA implicitfast装配失败:{status}')
        return out[:nv].copy(),out[nv:nv+nv*nv].reshape(nv,nv).copy(),out[nv+nv*nv:].copy()

    def step(self,inputs,*,warm_start=None,time=0.):
        with self._lock:
            if self._closed:raise RuntimeError('CUDA implicitfast工作区已关闭')
            return self._step_locked(inputs,warm_start=warm_start,time=time)

    def _step_locked(self,inputs,*,warm_start=None,time=0.):
        if self.contact_provider is None:raise ValueError('必须显式提供contact_provider，或调用step_with_contacts')
        self.layout.validate_input(inputs)
        contacts,pair_ids=self.contact_provider(inputs.state,inputs.palm)
        return self.step_with_contacts(inputs,contacts,pair_ids,warm_start=warm_start,time=time)

    def step_with_contacts(self,inputs,contacts,pair_ids,*,warm_start=None,time=0.):
        with self._lock:
            if self._closed:raise RuntimeError('CUDA implicitfast工作区已关闭')
            return self._step_contacts_locked(inputs,contacts,pair_ids,warm_start=warm_start,time=time)

    def _step_contacts_locked(self,inputs,contacts,pair_ids,*,warm_start=None,time=0.):
        if not isinstance(inputs,HandInput):raise TypeError('需要HandInput')
        torque=self.layout.validate_input(inputs);time=float(time)
        if not np.isfinite(time) or not np.isfinite(time+self.h):raise ValueError('time必须有限')
        state,palm=inputs.state,inputs.palm;nv=self.layout.nv
        M=self.numeric.finite(self.dynamics.mass_matrix(state,palm),'质量矩阵')
        bias=self.dynamics.inverse(state,palm,np.zeros(nv)).effort
        effort=np.zeros(nv);effort[:self.layout.hand_dofs]=torque
        kv=np.zeros(nv)
        if inputs.command.mode=='position_pd':kv[:self.layout.hand_dofs]=inputs.command.kv
        kv=self.numeric.finite(kv,'PD速度增益')
        rhs,implicit,_=self._prepare(M,state,bias,effort,kv,np.zeros(nv))
        free=self.solver.linear_solve(M,rhs)
        table=self.tables.assemble(state,palm,contacts,pair_ids)
        newton=self.solver.solve(M,free,table,warm_start=warm_start)
        astar=newton.acceleration
        _,_,advance_rhs=self._prepare(M,state,bias,effort,kv,astar)
        advanced=self.solver.linear_solve(implicit,advance_rhs)
        residual=float(np.max(abs(implicit.astype(float)@advanced-advance_rhs)))
        scale=1+np.max(abs(advance_rhs))+np.max(np.sum(abs(implicit.astype(float)),axis=1))*np.max(abs(advanced))
        if residual>3e-5*scale:raise RuntimeError('implicitfast线性残差过大')
        result=self.integrator.integrate(state,advanced,self.h)
        warm=astar.copy();coeff=self.numeric.finite(self.damping+kv,'速度系数')
        for a in (warm,advanced,free,M,coeff):a.setflags(write=False)
        return ImplicitStepResult(result,time+self.h,warm,astar,advanced,free,M,coeff,table,newton,residual)
