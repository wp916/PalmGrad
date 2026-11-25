"""共用状态积分的解析VJP；不包含M/RNE/接触对位置的导数。

原quat_integrate_vjp逐根复用。qpos是四维四元数数组余切，
position_tangent是右乘局部旋转扰动的三维余切，不是角速度梯度。
"""
import ctypes
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import tempfile
import re
import numpy as np
from nablaengine.cuda import compile_cuda
from nablaengine.g1_engine import source
from nablaengine.g1_engine.adjoint import ADJ_DEVICE
from .numerics import Numerics
from .time_step import CudaStateIntegrator


def _original(name):
    match=re.search(r'__device__ (?:__forceinline__ )?[^\n]+?\b'+name+r'\(',ADJ_DEVICE)
    if match is None:raise RuntimeError(f'原反向函数缺失:{name}')
    start=ADJ_DEVICE.index('{',match.end());end=start+1;depth=1
    while depth:
        depth+=(ADJ_DEVICE[end]=='{')-(ADJ_DEVICE[end]=='}');end+=1
    return ADJ_DEVICE[match.start():end]


CUDA=r'''
__global__ void pullback(float h,const float* input,const int* meta,float* out){
 const float* q=input;const float* v=q+NQ;const float* a=v+NV;
 const float* qb=a+NV;const float* vb=qb+NQ;
 float* oq=out;float* ov=oq+NQ;float* oa=ov+NV;float* ot=oa+NV;
 for(int i=0;i<NQ;i++)oq[i]=qb[i];
 for(int i=0;i<NV;i++){ov[i]=vb[i];ot[i]=0;}
 for(int b=1;b<NB;b++){
  int kind=meta[3*NB+b],qi=meta[NB+b],vi=meta[2*NB+b];
  if(kind==2){ov[vi]+=h*qb[qi];ot[vi]=qb[qi];}
  else if(kind==3){
   for(int k=0;k<3;k++){ov[vi+k]+=h*qb[qi+k];ot[vi+k]=qb[qi+k];}
   float w[3],gq[4],gw[3];
   for(int k=0;k<3;k++)w[k]=fmaf(h,a[vi+3+k],v[vi+3+k]);
   quat_integrate_vjp(q+qi+3,w,h,qb+qi+3,gq,gw);
   for(int k=0;k<4;k++)oq[qi+3+k]=gq[k];
   for(int k=0;k<3;k++)ov[vi+3+k]+=gw[k];
   // dq = .5*q*(0,dtheta)，dtheta采用物体局部坐标（右乘）。
   const float* r=q+qi+3;
   ot[vi+3]=.5f*(-r[1]*gq[0]+r[0]*gq[1]+r[3]*gq[2]-r[2]*gq[3]);
   ot[vi+4]=.5f*(-r[2]*gq[0]-r[3]*gq[1]+r[0]*gq[2]+r[1]*gq[3]);
   ot[vi+5]=.5f*(-r[3]*gq[0]+r[2]*gq[1]-r[1]*gq[2]+r[0]*gq[3]);
  }
 }
 for(int i=0;i<NV;i++)oa[i]=h*ov[i];
}
extern "C" int run(float h,const float* input,const int* meta,float* out){
 float *di=nullptr,*dout=nullptr;int* dm=nullptr;cudaError_t e;
 #define ICHECK(call) if((e=(call))!=cudaSuccess){cudaFree(di);cudaFree(dout);cudaFree(dm);return(int)e;}
 ICHECK(cudaMalloc(&di,(2*NQ+3*NV)*sizeof(float)));
 ICHECK(cudaMalloc(&dout,(NQ+3*NV)*sizeof(float)));
 ICHECK(cudaMalloc(&dm,4*NB*sizeof(int)));
 ICHECK(cudaMemcpy(di,input,(2*NQ+3*NV)*sizeof(float),cudaMemcpyHostToDevice));
 ICHECK(cudaMemcpy(dm,meta,4*NB*sizeof(int),cudaMemcpyHostToDevice));
 pullback<<<1,1>>>(h,di,dm,dout);ICHECK(cudaGetLastError());ICHECK(cudaDeviceSynchronize());
 ICHECK(cudaMemcpy(out,dout,(NQ+3*NV)*sizeof(float),cudaMemcpyDeviceToHost));
 cudaFree(di);cudaFree(dout);cudaFree(dm);return 0;
}
'''


@dataclass(frozen=True)
class IntegrationVJP:
    qpos: np.ndarray
    qvel: np.ndarray
    qacc: np.ndarray
    position_tangent: np.ndarray
    convention: str = 'integrator_only_local_right_rotation_tangent'


class CudaIntegrationAdjoint:
    """固定积分加速度的输入状态VJP，以及独立的加速度VJP。

    evaluate(state,qacc,dt,qpos_next_bar,qvel_next_bar)。不调用动力学；
    position_tangent按canonical nv排列，平移世界坐标、旋转物体局部坐标。
    """
    def __init__(self,dynamics,*,numerics='float32'):
        self.dynamics=dynamics;self.layout=dynamics.layout;self.numeric=Numerics(numerics)
        m=SimpleNamespace(nbody=dynamics.motion.nb,nq=self.layout.nq,nv=self.layout.nv)
        helpers='\n'.join(_original(n) for n in ('dot4','adj_sinc','adj_dsinc_x','quat_integrate_vjp'))
        code=source.header(m,1,1)+source.DEVICE+helpers+CUDA
        with tempfile.TemporaryDirectory(prefix='nabla_integration_adjoint_') as tmp:
            cu,so=Path(tmp)/'adj.cu',Path(tmp)/'adj.so';cu.write_text(self.numeric.cuda(code))
            compile_cuda(cu,so,extra=('-shared','-Xcompiler','-fPIC'));self._library=ctypes.CDLL(str(so))
        self._run=self._library.run;fp=self.numeric.pointer
        self._run.argtypes=[self.numeric.scalar,fp,ctypes.POINTER(ctypes.c_int),fp];self._run.restype=ctypes.c_int

    def evaluate(self,state,qacc,dt,qpos_next_bar,qvel_next_bar):
        self.layout.validate_state(state);dt=CudaStateIntegrator._dt(dt)
        arrays=[]
        for value,n,name in ((qacc,self.layout.nv,'qacc'),(qpos_next_bar,self.layout.nq,'qpos_next_bar'),
                             (qvel_next_bar,self.layout.nv,'qvel_next_bar')):
            x=self.numeric.finite(value,name)
            if x.shape!=(n,):raise ValueError(f'{name}形状必须为({n},)')
            arrays.append(x)
        a,qb,vb=arrays
        packed=self.numeric.finite(np.r_[state.qpos,state.qvel,a,qb,vb],'积分反向输入')
        nq,nv=self.layout.nq,self.layout.nv;out=np.empty(nq+3*nv,self.numeric.dtype)
        fp=self.numeric.pointer;ip=ctypes.POINTER(ctypes.c_int)
        status=self._run(dt,packed.ctypes.data_as(fp),self.dynamics.motion.meta.ctypes.data_as(ip),out.ctypes.data_as(fp))
        if status or not np.isfinite(out).all():raise RuntimeError(f'CUDA积分反向失败:{status}')
        out.setflags(write=False)
        return IntegrationVJP(out[:nq],out[nq:nq+nv],out[nq+nv:nq+2*nv],out[nq+2*nv:])
