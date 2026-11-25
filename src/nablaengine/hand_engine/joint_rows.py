"""hinge限位和frictionloss约束行；沿用原kbimp/efc_D，显式hand索引。"""
import ctypes
from dataclasses import dataclass
from pathlib import Path
import tempfile
import numpy as np
import mujoco
from nablaengine.cuda import compile_cuda
from .constraint_rows import CUDA as SOFT_CUDA
from .numerics import Numerics


@dataclass(frozen=True)
class JointRows:
    jacobian: np.ndarray
    velocity: np.ndarray
    D: np.ndarray
    aref: np.ndarray
    frictionloss: np.ndarray
    kind: np.ndarray
    a: np.ndarray
    b: np.ndarray
    damping: np.ndarray


CUDA=SOFT_CUDA.split('__global__ void normal_rows')[0]+r'''
__global__ void joint_rows(int n,float h,const float* material,const float* state,float* out){
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
  const float* m=material+19*i;float q=state[i],v=state[n+i];
  float lo=RSUB(q,m[0]),hi=RSUB(m[1],q);
  for(int limit=0;limit<2;limit++){
    float* y=out+(limit*n+i)*10;
    float sign=limit?(lo<hi?1.f:-1.f):1.f;
    float pos=limit?fminf(lo,hi):0.f;
    y[0]=limit?(m[2]!=0.f && pos<0.f):(m[3]>0.f);
    y[1]=sign;y[2]=pos;y[3]=sign*v;
    float k,b,imp;
    kbimp(m+(limit?5:12),m+(limit?7:14),pos,h,&k,&b,&imp);
    y[4]=k;y[5]=b;y[6]=imp;y[7]=efc_D(m[4],imp);
    y[8]=-k*imp*pos-b*y[3];y[9]=limit?0.f:m[3];
  }
}
extern "C" int run(int n,float h,const float* material,const float* state,float* output){
  float *dm=nullptr,*ds=nullptr,*dout=nullptr;cudaError_t e;
  #define JRCHECK(call) if((e=(call))!=cudaSuccess){cudaFree(dm);cudaFree(ds);cudaFree(dout);return(int)e;}
  JRCHECK(cudaMalloc(&dm,n*19*sizeof(float)));JRCHECK(cudaMalloc(&ds,n*2*sizeof(float)));
  JRCHECK(cudaMalloc(&dout,n*20*sizeof(float)));
  JRCHECK(cudaMemcpy(dm,material,n*19*sizeof(float),cudaMemcpyHostToDevice));
  JRCHECK(cudaMemcpy(ds,state,n*2*sizeof(float),cudaMemcpyHostToDevice));
  joint_rows<<<(n+31)/32,32>>>(n,h,dm,ds,dout);JRCHECK(cudaGetLastError());JRCHECK(cudaDeviceSynchronize());
  JRCHECK(cudaMemcpy(output,dout,n*20*sizeof(float),cudaMemcpyDeviceToHost));
  cudaFree(dm);cudaFree(ds);cudaFree(dout);return 0;
}
'''


class CudaJointRows:
    """当前仅hinge frictionloss，零joint margin，正solref，refsafe开启。

    每关节保留自己的材料参数，不照搬G1所有关节共用一份材料的特化。
    frictionloss是有界干摩擦约束，不是被动力矩damping，也不是接触滑动摩擦。
    """
    def __init__(self,model,plan,*,numerics="float32"):
        self.numeric=Numerics(numerics)
        if int(model.opt.disableflags)&int(mujoco.mjtDisableBit.mjDSBL_REFSAFE):
            raise ValueError('当前要求refsafe开启')
        self.plan=plan;self.layout=plan.layout;self.n=self.layout.hand_dofs
        blocks=[b for b in self.layout.blocks if b.kind=='hinge']
        used={b.source_qvel_start for b in blocks}
        if any(model.dof_frictionloss[i]!=0 for i in range(model.nv) if i not in used):
            raise ValueError('当前只支持手指hinge的frictionloss，free物体必须为零')
        data=[]
        for b in blocks:
            j,d=b.joint_id,b.source_qvel_start
            if model.jnt_margin[j]!=0:raise ValueError('当前要求hinge margin=0')
            data.append(np.r_[model.jnt_range[j],model.jnt_limited[j],model.dof_frictionloss[d],model.dof_invweight0[d],
                              model.jnt_solref[j],model.jnt_solimp[j],model.dof_solref[d],model.dof_solimp[d]])
        self.material=self.numeric.finite(np.asarray(data).reshape(self.n,19),'关节材料')
        if np.any(self.material[:,3:5]<0) or np.any(self.material[:,5:7]<=0) or np.any(self.material[:,12:14]<=0):
            raise ValueError('frictionloss/invweight必须非负，solref必须为正')
        self.material.setflags(write=False)
        with tempfile.TemporaryDirectory(prefix='nabla_hand_joint_rows_') as tmp:
            cu,so=Path(tmp)/'joint.cu',Path(tmp)/'joint.so';cu.write_text(self.numeric.cuda(CUDA))
            compile_cuda(cu,so,extra=('-shared','-Xcompiler','-fPIC'));self._library=ctypes.CDLL(str(so))
        self._run=self._library.run;fp=self.numeric.pointer
        self._run.argtypes=[ctypes.c_int,self.numeric.scalar,fp,fp,fp];self._run.restype=ctypes.c_int

    def assemble(self,state):
        self.layout.validate_state(state);n=self.n
        packed=self.numeric.finite(np.r_[state.qpos[:n],state.qvel[:n]],'关节状态')
        output=np.empty((2*n,10),self.numeric.dtype)
        if n:
            fp=self.numeric.pointer
            status=self._run(n,self.plan.physics['timestep'],self.material.ctypes.data_as(fp),packed.ctypes.data_as(fp),output.ctypes.data_as(fp))
            if status:raise RuntimeError(f'CUDA关节行失败，错误码{status}')
        if not np.isfinite(output).all():raise RuntimeError('CUDA关节行产生非有限输出')
        active=np.flatnonzero(output[:,0]);rows=output[active];dof=active%n
        self.plan.check_counts(0,len(rows))
        J=np.zeros((len(rows),self.layout.nv),self.numeric.dtype);J[np.arange(len(rows)),dof]=rows[:,1]
        parts=[J,rows[:,3].copy(),rows[:,7].copy(),rows[:,8].copy(),rows[:,9].copy(),
               (active>=n).astype(np.uint8),dof.astype(np.uint16),rows[:,1].astype(np.int8),rows[:,5].copy()]
        for p in parts:p.setflags(write=False)
        return JointRows(*parts)
