"""固定约束分支单步的速度/动作VJP；输入位置梯度仍未实现。"""
from dataclasses import dataclass
import numpy as np
import ctypes,tempfile
from pathlib import Path
from nablaengine.cuda import compile_cuda
from .action_adjoint import CudaActionAdjoint,ActionTape
from .rne_velocity import CudaRNEVelocityAdjoint


@dataclass(frozen=True)
class StepVelocityVJP:
    qvel: np.ndarray
    action: np.ndarray
    convention: str = 'fixed_branch_step_velocity_vjp_fixed_q'



CUDA=r'''
#include <cuda_runtime.h>
__global__ void transpose_rows(int nr,const double* J,const double* weights,const double* lam,double* out){
 int i=threadIdx.x;for(;i<NV;i+=32){double sum=0;
 for(int r=0;r<nr;r++){double dot=0;for(int j=0;j<NV;j++)dot+=J[r*NV+j]*lam[j];sum+=J[r*NV+i]*weights[r]*dot;}out[i]=sum;}
}
extern "C" int run(int nr,const double* J,const double* weights,const double* lam,double* out){
 double *dj=nullptr,*dw=nullptr,*dl=nullptr,*do_=nullptr;cudaError_t e;
 #define CHECK(call) if((e=(call))!=cudaSuccess){cudaFree(dj);cudaFree(dw);cudaFree(dl);cudaFree(do_);return(int)e;}
 CHECK(cudaMalloc(&dj,max(1,nr)*NV*sizeof(double)));CHECK(cudaMalloc(&dw,max(1,nr)*sizeof(double)));
 CHECK(cudaMalloc(&dl,NV*sizeof(double)));CHECK(cudaMalloc(&do_,NV*sizeof(double)));
 if(nr){CHECK(cudaMemcpy(dj,J,nr*NV*sizeof(double),cudaMemcpyHostToDevice));CHECK(cudaMemcpy(dw,weights,nr*sizeof(double),cudaMemcpyHostToDevice));}
 CHECK(cudaMemcpy(dl,lam,NV*sizeof(double),cudaMemcpyHostToDevice));transpose_rows<<<1,32>>>(nr,dj,dw,dl,do_);
 CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());CHECK(cudaMemcpy(out,do_,NV*sizeof(double),cudaMemcpyDeviceToHost));
 cudaFree(dj);cudaFree(dw);cudaFree(dl);cudaFree(do_);return 0;
}
'''

class CudaVelocityAdjoint:
    """固定q/palm/PD增益，使用前向末次约束分支，对输入速度与动作求导。

    支持法向、金字塔摩擦、hinge限位/干摩擦。完整多步反向仍缺位置导数。
    """
    def __init__(self,engine):
        self.engine=engine;self.action=CudaActionAdjoint(engine)
        self.rne=CudaRNEVelocityAdjoint(engine.dynamics)
        with tempfile.TemporaryDirectory(prefix='nabla_velocity_rows_') as tmp:
            cu,so=Path(tmp)/'rows.cu',Path(tmp)/'rows.so'
            cu.write_text(f'#define NV {engine.layout.nv}\n'+CUDA)
            compile_cuda(cu,so,extra=('-shared','-Xcompiler','-fPIC'));self._library=ctypes.CDLL(str(so))
        self._run=self._library.run;p=ctypes.POINTER(ctypes.c_double)
        self._run.argtypes=[ctypes.c_int,p,p,p,p];self._run.restype=ctypes.c_int

    def record(self,inputs,**kwargs):
        return self.action.record(inputs,**kwargs)

    def vjp(self,tape,qpos_bar,qvel_bar):
        if not isinstance(tape,ActionTape) or tape._owner is not self.action._owner:
            raise ValueError('tape不属于此速度反向实例')
        with self.engine._lock:
            pull=self.action.vjp(tape,qpos_bar,qvel_bar)
            r=tape.forward;table=r.constraints;n=self.engine.layout.nv
            if table.velocity_damping is None:raise ValueError('前向记录缺少约束行速度阻尼系数')
            lam=self.engine.solver.linear_solve(r.newton.evaluation.hessian,pull.acceleration_cotangent)
            inverse=self.rne.evaluate(tape.inputs.state,tape.inputs.palm,lam)
            J=np.ascontiguousarray(table.jacobian,dtype=np.float64)
            weights=np.ascontiguousarray(table.velocity_damping*r.newton.evaluation.row_curvature,dtype=np.float64)
            vector=np.ascontiguousarray(lam,dtype=np.float64);constraint=np.empty(n,np.float64)
            p=ctypes.POINTER(ctypes.c_double)
            status=self._run(len(weights),J.ctypes.data_as(p),weights.ctypes.data_as(p),vector.ctypes.data_as(p),constraint.ctypes.data_as(p))
            if status or not np.isfinite(constraint).all():raise RuntimeError(f'CUDA约束速度反传失败:{status}')
            # 固定q：dg/dv=db/dv+C+Kv+J.T*diag(curvature*damping)*J。
            grad=self.engine.numeric.finite(pull.advancement_cotangent/self.engine.h-
                    r.velocity_coefficient*lam-inverse.qvel_bar-constraint,'输入速度梯度')
            grad.setflags(write=False);return StepVelocityVJP(grad,pull.values)
