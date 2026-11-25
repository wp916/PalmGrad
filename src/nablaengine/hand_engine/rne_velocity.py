"""固定q、qacc、PalmMotion时的逆动力学速度Jacobian/VJP。

解析线性化原hinge运动递推和body_force，再复用原子树累加/关节投影。
不做速度差分，不调用MuJoCo；不是完整step状态反向，也不包括被动阻尼/接触。
"""
import ctypes
from dataclasses import dataclass
from pathlib import Path
import tempfile
from types import SimpleNamespace
import numpy as np
from nablaengine.cuda import compile_cuda
from nablaengine.g1_engine import source
from nablaengine.model.hand import PalmMotion
from .action_adjoint import _original


CUDA=r'''
__device__ __constant__ int rv_kind[NB];
__global__ void velocity_columns(const float* motion,const float* velocity,const float* bar,float* output,float* pullback){
 int col=blockIdx.x;
 float X[NB][3]={},Q[NB][4]={},dW[NB][3]={},dAL[NB][3]={},dA[NB][3]={};
 float F[NB][3]={},N[NB][3]={},effort[NV]={};
 Q[0][0]=1;
 for(int b=1;b<NB;b++){
   const float* m=motion+19*b;int p=c_parent[b],vi=c_dadr[b];
   for(int k=0;k<3;k++)X[b][k]=m[k];for(int k=0;k<4;k++)Q[b][k]=m[k+3];
   if(rv_kind[b]==2){
     const float* Wp=motion+19*p+7;float r[3],axis[3],awd[3],dawd[3],a[3],c[3],d[3],wr[3],dwr[3];
     qrot_rn(Q[b],c_axis+3*b,axis);float dv=col==vi?1:0;
     for(int k=0;k<3;k++){
       r[k]=X[b][k]-X[p][k];awd[k]=axis[k]*velocity[vi];dawd[k]=axis[k]*dv;dW[b][k]=dW[p][k]+dawd[k];
     }
     cross(dW[p],awd,a);cross(Wp,dawd,c);
     for(int k=0;k<3;k++)dAL[b][k]=dAL[p][k]+a[k]+c[k];
     cross(Wp,r,wr);cross(dW[p],r,dwr);cross(dAL[p],r,a);cross(dW[p],wr,c);cross(Wp,dwr,d);
     for(int k=0;k<3;k++)dA[b][k]=dA[p][k]+a[k]+c[k]+d[k];
   }else if(rv_kind[b]==3 && col>=vi+3 && col<vi+6){
     float unit[3]={};unit[col-vi-3]=1;qrot_rn(Q[b],unit,dW[b]);
   }
   // dF=m[dA+dAL×c+dW×(W×c)+W×(dW×c)]。
   const float* W=m+7;float c[3],wc[3],dwc[3],u[3],v[3],w[3],qi[4],Iw[3],Idw[3],Idal[3],cf[3];
   qrot(Q[b],c_ipos+3*b,c);cross(W,c,wc);cross(dW[b],c,dwc);
   cross(dAL[b],c,u);cross(dW[b],wc,v);cross(W,dwc,w);
   for(int k=0;k<3;k++)F[b][k]=c_mass[b]*(dA[b][k]+u[k]+v[k]+w[k]);
   // dN=I*dAL+dW×IW+W×I*dW+c×dF；固定q，因此dI=dc=0。
   qmul(Q[b],c_iquat+4*b,qi);inert_mul(qi,c_inert+3*b,W,Iw);
   inert_mul(qi,c_inert+3*b,dW[b],Idw);inert_mul(qi,c_inert+3*b,dAL[b],Idal);
   cross(dW[b],Iw,u);cross(W,Idw,v);cross(c,F[b],cf);
   for(int k=0;k<3;k++)N[b][k]=Idal[k]+u[k]+v[k]+cf[k];
 }
 for(int b=NB-1;b>=1;b--){
   int vi=c_dadr[b];
   if(rv_kind[b]==2)hinge_torque(b,Q[b],N[b],effort);
   else if(rv_kind[b]==3){float local[3];qrot_inv(Q[b],N[b],local);
     for(int k=0;k<3;k++){effort[vi+k]=F[b][k];effort[vi+3+k]=local[k];}}
   int p=c_parent[b];if(p>0)accumulate(X[p],X[b],F[b],N[b],F[p],N[p]);
 }
 float dot=0;for(int i=0;i<NV;i++){output[i*NV+col]=effort[i];dot+=bar[i]*effort[i];}pullback[col]=dot;
}
extern "C" int run(const float* motion,const float* velocity,const float* bar,const int* meta,const float* constants,float* output,float* pullback){
 float *dm=nullptr,*dv=nullptr,*db=nullptr,*do_=nullptr,*dp=nullptr;cudaError_t e;
 #define VC(call) if((e=(call))!=cudaSuccess){cudaFree(dm);cudaFree(dv);cudaFree(db);cudaFree(do_);cudaFree(dp);return(int)e;}
 VC(cudaMemcpyToSymbol(c_parent,meta,NB*sizeof(int)));VC(cudaMemcpyToSymbol(c_dadr,meta+2*NB,NB*sizeof(int)));
 VC(cudaMemcpyToSymbol(rv_kind,meta+3*NB,NB*sizeof(int)));
 const float* c=constants;
 VC(cudaMemcpyToSymbol(c_axis,c,3*NB*sizeof(float)));c+=3*NB;
 VC(cudaMemcpyToSymbol(c_ipos,c,3*NB*sizeof(float)));c+=3*NB;
 VC(cudaMemcpyToSymbol(c_iquat,c,4*NB*sizeof(float)));c+=4*NB;
 VC(cudaMemcpyToSymbol(c_mass,c,NB*sizeof(float)));c+=NB;
 VC(cudaMemcpyToSymbol(c_inert,c,3*NB*sizeof(float)));
 VC(cudaMalloc(&dm,NB*19*sizeof(float)));VC(cudaMalloc(&dv,NV*sizeof(float)));VC(cudaMalloc(&db,NV*sizeof(float)));
 VC(cudaMalloc(&do_,NV*NV*sizeof(float)));VC(cudaMalloc(&dp,NV*sizeof(float)));
 VC(cudaMemcpy(dm,motion,NB*19*sizeof(float),cudaMemcpyHostToDevice));VC(cudaMemcpy(dv,velocity,NV*sizeof(float),cudaMemcpyHostToDevice));
 VC(cudaMemcpy(db,bar,NV*sizeof(float),cudaMemcpyHostToDevice));velocity_columns<<<NV,1>>>(dm,dv,db,do_,dp);
 VC(cudaGetLastError());VC(cudaDeviceSynchronize());VC(cudaMemcpy(output,do_,NV*NV*sizeof(float),cudaMemcpyDeviceToHost));
 VC(cudaMemcpy(pullback,dp,NV*sizeof(float),cudaMemcpyDeviceToHost));
 cudaFree(dm);cudaFree(dv);cudaFree(db);cudaFree(do_);cudaFree(dp);return 0;
}
'''


@dataclass(frozen=True)
class RNEVelocityResult:
    jacobian: np.ndarray
    qvel_bar: np.ndarray


class CudaRNEVelocityAdjoint:
    """输出d(inverse_effort)/d(qvel)及其转置乘effort_bar；逐列GPU正确性实现。"""
    def __init__(self,dynamics):
        self.dynamics=dynamics;self.layout=dynamics.layout;self.numeric=dynamics.numeric
        m=SimpleNamespace(nbody=dynamics.motion.nb,nq=self.layout.nq,nv=self.layout.nv)
        code=source.header(m,1,1)+source.DEVICE+_original('inert_mul')+CUDA
        with tempfile.TemporaryDirectory(prefix='nabla_rne_velocity_') as tmp:
            cu,so=Path(tmp)/'velocity.cu',Path(tmp)/'velocity.so';cu.write_text(self.numeric.cuda(code))
            compile_cuda(cu,so,extra=('-shared','-Xcompiler','-fPIC'));self._library=ctypes.CDLL(str(so))
        self._run=self._library.run;p=self.numeric.pointer;ip=ctypes.POINTER(ctypes.c_int)
        self._run.argtypes=[p,p,p,ip,p,p,p];self._run.restype=ctypes.c_int

    def evaluate(self,state,palm,effort_bar,*,qacc=None):
        self.layout.validate_state(state)
        if not isinstance(palm,PalmMotion):raise TypeError('需要显式PalmMotion')
        n=self.layout.nv;bar=self.numeric.finite(effort_bar,'effort_bar')
        if bar.shape!=(n,):raise ValueError('effort_bar长度必须为nv')
        a=np.zeros(n) if qacc is None else self.numeric.finite(qacc,'qacc')
        motion=self.dynamics.motion.forward(state,palm,a)
        packed=self.numeric.finite(np.c_[motion.position,motion.quaternion,motion.angular_velocity,motion.linear_velocity,
                                        motion.angular_acceleration,motion.linear_acceleration],'刚体运动')
        v=self.numeric.finite(state.qvel,'qvel');J=np.empty((n,n),self.numeric.dtype);g=np.empty(n,self.numeric.dtype)
        p=self.numeric.pointer;ip=ctypes.POINTER(ctypes.c_int)
        code=self._run(packed.ctypes.data_as(p),v.ctypes.data_as(p),bar.ctypes.data_as(p),self.dynamics.motion.meta.ctypes.data_as(ip),
                       self.dynamics.constants.ctypes.data_as(p),J.ctypes.data_as(p),g.ctypes.data_as(p))
        if code or not np.isfinite(J).all() or not np.isfinite(g).all():raise RuntimeError(f'CUDA RNE速度反向失败:{code}')
        J.setflags(write=False);g.setflags(write=False);return RNEVelocityResult(J,g)
