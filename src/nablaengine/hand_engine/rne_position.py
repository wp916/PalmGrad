"""逆动力学对位置局部扰动的解析CUDA Jacobian/VJP。

固定canonical速度/加速度和规定手掌；自由物体旋转采用右乘局部扰动。
不包含接触、被动项或完整step的位置反向。逐列基线，非融合批量实现。
"""
import ctypes
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import tempfile
import numpy as np
from nablaengine.cuda import compile_cuda
from nablaengine.g1_engine import source
from nablaengine.model.hand import HandState,PalmMotion
from .integration_adjoint import _original
from .rne_velocity import CUDA as VELOCITY_CUDA


CUDA=r'''
__device__ __constant__ int rv_kind[NB];
// 固定向量x时，旋转惯量张量的变化 dI*x=theta×(I*x)-I*(theta×x)。
__device__ void inertia_delta(const float* qi,const float* inertia,const float* theta,const float* x,float* out){
 float ix[3],tx[3],itx[3];inert_mul(qi,inertia,x,ix);cross(theta,x,tx);inert_mul(qi,inertia,tx,itx);
 cross(theta,ix,out);for(int k=0;k<3;k++)out[k]-=itx[k];
}
__global__ void velocity_columns(const float* motion,const float* inputs,const float* bar,float* output,float* pullback){
 int col=blockIdx.x;const float* gravity=inputs;const float* velocity=inputs+3;const float* acceleration=velocity+NV;
 float X[NB][3]={},Q[NB][4]={},T[NB][3]={},dX[NB][3]={},dW[NB][3]={},dAL[NB][3]={},dA[NB][3]={};
 float F[NB][3]={},N[NB][3]={},dF[NB][3]={},dN[NB][3]={},effort[NV]={};Q[0][0]=1;
 for(int b=1;b<NB;b++){
  const float* m=motion+19*b;int p=c_parent[b],vi=c_dadr[b];
  for(int k=0;k<3;k++)X[b][k]=m[k];for(int k=0;k<4;k++)Q[b][k]=m[k+3];
  float r[3],dr[3],u[3],v[3],w[3],z[3];
  for(int k=0;k<3;k++)r[k]=X[b][k]-X[p][k];
  if(rv_kind[b]==2){
   qrot_rn(Q[b],c_axis+3*b,z);cross(T[p],r,u);
   for(int k=0;k<3;k++){T[b][k]=T[p][k]+(col==vi?z[k]:0);dX[b][k]=dX[p][k]+u[k];}
   // W_b=W_p+z*vj：theta_b×W_b不能代替父节点速度的独立变化。
   float jointW[3],jointAL[3],djointW[3],djointAL[3],Wp[3],ALp[3];
   // 直接使用输入关节速度/加速度，避免从两个大世界量相减恢复小关节量。
   for(int k=0;k<3;k++){Wp[k]=motion[19*p+7+k];ALp[k]=motion[19*p+13+k];jointW[k]=z[k]*velocity[vi];jointAL[k]=z[k]*acceleration[vi];}
   cross(T[b],jointW,djointW);cross(T[b],jointAL,djointAL);
   cross(dW[p],jointW,u);cross(Wp,djointW,v);
   for(int k=0;k<3;k++){dW[b][k]=dW[p][k]+djointW[k];dAL[b][k]=dAL[p][k]+u[k]+v[k]+djointAL[k];dr[k]=dX[b][k]-dX[p][k];}
   float wr[3],dwr[3];cross(Wp,r,wr);cross(dW[p],r,dwr);cross(Wp,dr,u);
   for(int k=0;k<3;k++)dwr[k]+=u[k];
   cross(dAL[p],r,u);cross(ALp,dr,v);cross(dW[p],wr,w);cross(Wp,dwr,z);
   for(int k=0;k<3;k++)dA[b][k]=dA[p][k]+u[k]+v[k]+w[k]+z[k];
  }else if(rv_kind[b]==3){
   if(col>=vi && col<vi+3)dX[b][col-vi]=1;
   if(col>=vi+3 && col<vi+6){float unit[3]={};unit[col-vi-3]=1;qrot_rn(Q[b],unit,T[b]);}
   cross(T[b],m+7,dW[b]);cross(T[b],m+13,dAL[b]);
  }
  float A[3];for(int k=0;k<3;k++)A[k]=m[16+k]-gravity[k];
  body_force(b,Q[b],m+7,m+13,A,c_ipos+3*b,F[b],N[b]);
  float c[3],dc[3],wc[3],dwc[3],qi[4],Iw[3],Idw[3],Idal[3],dIw[3],dIal[3];
  qrot(Q[b],c_ipos+3*b,c);cross(T[b],c,dc);cross(m+7,c,wc);
  cross(dW[b],c,dwc);cross(m+7,dc,u);for(int k=0;k<3;k++)dwc[k]+=u[k];
  cross(dAL[b],c,u);cross(m+13,dc,v);cross(dW[b],wc,w);cross(m+7,dwc,z);
  for(int k=0;k<3;k++)dF[b][k]=c_mass[b]*(dA[b][k]+u[k]+v[k]+w[k]+z[k]);
  qmul(Q[b],c_iquat+4*b,qi);inert_mul(qi,c_inert+3*b,m+7,Iw);
  inert_mul(qi,c_inert+3*b,dW[b],Idw);inert_mul(qi,c_inert+3*b,dAL[b],Idal);
  inertia_delta(qi,c_inert+3*b,T[b],m+7,dIw);inertia_delta(qi,c_inert+3*b,T[b],m+13,dIal);
  for(int k=0;k<3;k++)dIw[k]+=Idw[k];
  cross(dW[b],Iw,u);cross(m+7,dIw,v);cross(dc,F[b],w);cross(c,dF[b],z);
  for(int k=0;k<3;k++)dN[b][k]=dIal[k]+Idal[k]+u[k]+v[k]+w[k]+z[k];
 }
 for(int b=NB-1;b>=1;b--){
  int vi=c_dadr[b];
  if(rv_kind[b]==2){float axis[3],da[3];qrot_rn(Q[b],c_axis+3*b,axis);cross(T[b],axis,da);effort[vi]=dot3(da,N[b])+dot3(axis,dN[b]);}
  else if(rv_kind[b]==3){float t[3],local[3];cross(T[b],N[b],t);for(int k=0;k<3;k++)t[k]=dN[b][k]-t[k];qrot_inv(Q[b],t,local);
   for(int k=0;k<3;k++){effort[vi+k]=dF[b][k];effort[vi+3+k]=local[k];}}
  int p=c_parent[b];if(p>0){
   float r[3],dr[3],u[3],v[3];for(int k=0;k<3;k++){r[k]=X[b][k]-X[p][k];dr[k]=dX[b][k]-dX[p][k];}
   cross(dr,F[b],u);cross(r,dF[b],v);
   for(int k=0;k<3;k++){dF[p][k]+=dF[b][k];dN[p][k]+=dN[b][k]+u[k]+v[k];}
   accumulate(X[p],X[b],F[b],N[b],F[p],N[p]);
  }
 }
 float sum=0;for(int i=0;i<NV;i++){output[i*NV+col]=effort[i];sum+=bar[i]*effort[i];}pullback[col]=sum;
}
'''
# 共用CUDA分配/参数上传包装；第二输入改成世界重力、速度、加速度的打包数组。
WRAPPER=VELOCITY_CUDA[VELOCITY_CUDA.index('extern "C" int run('):]
if 'cudaMalloc(&dv,NV*sizeof(float))' not in WRAPPER or 'cudaMemcpy(dv,velocity,NV*sizeof(float)' not in WRAPPER:
    raise RuntimeError('RNE共用包装签名改变，需要重新审核位置反向的重力上传长度')
WRAPPER=WRAPPER.replace('cudaMalloc(&dv,NV*sizeof(float))','cudaMalloc(&dv,(3+2*NV)*sizeof(float))').replace('cudaMemcpy(dv,velocity,NV*sizeof(float)','cudaMemcpy(dv,velocity,(3+2*NV)*sizeof(float)')


@dataclass(frozen=True)
class RNEPositionResult:
    jacobian: np.ndarray
    position_tangent_bar: np.ndarray


class CudaRNEPositionAdjoint:
    """d ID(q,v,a)/d局部位置扰动；返回nv×nv Jacobian与转置乘积。"""
    def __init__(self,dynamics):
        self.dynamics=dynamics;self.layout=dynamics.layout;self.numeric=dynamics.numeric
        m=SimpleNamespace(nbody=dynamics.motion.nb,nq=self.layout.nq,nv=self.layout.nv)
        code=source.header(m,1,1)+source.DEVICE+_original('inert_mul')+CUDA+WRAPPER
        with tempfile.TemporaryDirectory(prefix='nabla_rne_position_') as tmp:
            cu,so=Path(tmp)/'position.cu',Path(tmp)/'position.so';cu.write_text(self.numeric.cuda(code))
            compile_cuda(cu,so,extra=('-shared','-Xcompiler','-fPIC'));self._library=ctypes.CDLL(str(so))
        self._run=self._library.run;p=self.numeric.pointer;ip=ctypes.POINTER(ctypes.c_int)
        self._run.argtypes=[p,p,p,ip,p,p,p];self._run.restype=ctypes.c_int

    def evaluate(self,state,palm,effort_bar,*,qacc=None,gravity=None):
        self.layout.validate_state(state)
        if not isinstance(palm,PalmMotion):raise TypeError('需要显式PalmMotion')
        n=self.layout.nv;bar=self.numeric.finite(effort_bar,'effort_bar')
        if bar.shape!=(n,):raise ValueError('effort_bar长度必须为nv')
        a=np.zeros(n) if qacc is None else self.numeric.finite(qacc,'qacc')
        g=self.numeric.finite(self.dynamics.gravity if gravity is None else gravity,'gravity')
        if g.shape!=(3,):raise ValueError('gravity必须为3维世界向量')
        motion=self.dynamics.motion.forward(state,palm,a)
        packed_input=self.numeric.finite(np.r_[g,state.qvel,a],'重力与运动输入')
        packed=self.numeric.finite(np.c_[motion.position,motion.quaternion,motion.angular_velocity,motion.linear_velocity,
                                        motion.angular_acceleration,motion.linear_acceleration],'刚体运动')
        J=np.empty((n,n),self.numeric.dtype);grad=np.empty(n,self.numeric.dtype);p=self.numeric.pointer;ip=ctypes.POINTER(ctypes.c_int)
        status=self._run(packed.ctypes.data_as(p),packed_input.ctypes.data_as(p),bar.ctypes.data_as(p),self.dynamics.motion.meta.ctypes.data_as(ip),
                         self.dynamics.constants.ctypes.data_as(p),J.ctypes.data_as(p),grad.ctypes.data_as(p))
        if status or not np.isfinite(J).all() or not np.isfinite(grad).all():raise RuntimeError(f'CUDA RNE位置反向失败:{status}')
        J.setflags(write=False);grad.setflags(write=False);return RNEPositionResult(J,grad)

    def mass_product(self,state,palm,vector,effort_bar):
        """固定vector和effort_bar，返回d[effort_bar.T M(q) vector]/d局部位置扰动。"""
        if not isinstance(palm,PalmMotion):raise TypeError('需要显式PalmMotion')
        return self.evaluate(HandState(state.qpos,np.zeros(self.layout.nv)),PalmMotion.fixed(palm.position,palm.quaternion),
                             effort_bar,qacc=vector,gravity=(0,0,0))
