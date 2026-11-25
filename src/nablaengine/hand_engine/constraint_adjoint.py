"""固定行分支的约束残差VJP及软参数距离导数；尚无碰撞几何反向。"""
import ctypes,re,tempfile
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from nablaengine.cuda import compile_cuda
from nablaengine.g1_engine.adjoint_contact import DEVICE
from .constraint_rows import CUDA as SOFT_CUDA
from .numerics import Numerics


def _contact_function(name):
    match=re.search(r'__device__ __forceinline__ void '+name+r'\(',DEVICE)
    if match is None:raise RuntimeError(f'原接触导数函数缺失:{name}')
    start=DEVICE.index('{',match.end());end=start+1;depth=1
    while depth:
        depth+=(DEVICE[end]=='{')-(DEVICE[end]=='}');end+=1
    return DEVICE[match.start():end]


CUDA=SOFT_CUDA.split('__global__ void normal_rows')[0]+_contact_function('kbimp_d')+_contact_function('efc_D_d')+r'''
__global__ void soft(int nr,float h,const float* input,float* output){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=nr)return;
 const float* x=input+9*i;float k,b,imp,di,D,dD;
 kbimp_d(x,x+2,x[7],h,&k,&b,&imp,&di);efc_D_d(x[8],imp,di,&D,&dD);
 output[3*i]=di;output[3*i+1]=dD;output[3*i+2]=-k*(imp+x[7]*di);
}
__global__ void row_vjp(int nr,const float* rows,const float* a,const float* lam,float* output){
 int r=blockIdx.x*blockDim.x+threadIdx.x;if(r>=nr)return;
 const float* x=rows+r*(NV+4);float* o=output+r*(NV+2);float s=0;
 for(int j=0;j<NV;j++)s+=x[j]*lam[j];
 float residual=x[NV],ds=-x[NV+1],curvature=x[NV+2];int zone=(int)x[NV+3];
 for(int j=0;j<NV;j++)o[j]=ds*lam[j]+s*curvature*a[j];
 o[NV]=zone==1?s*residual:0;o[NV+1]=-s*curvature;
}
extern "C" int run(int mode,int nr,float h,const float* input,const float* a,const float* lam,float* output){
 float *di=nullptr,*da=nullptr,*dl=nullptr,*do_=nullptr;cudaError_t e;
 #define RC(call) if((e=(call))!=cudaSuccess){cudaFree(di);cudaFree(da);cudaFree(dl);cudaFree(do_);return(int)e;}
 int ni=mode==0?9:NV+4,no=mode==0?3:NV+2;
 RC(cudaMalloc(&di,nr*ni*sizeof(float)));RC(cudaMalloc(&do_,nr*no*sizeof(float)));
 RC(cudaMalloc(&da,NV*sizeof(float)));RC(cudaMalloc(&dl,NV*sizeof(float)));
 RC(cudaMemcpy(di,input,nr*ni*sizeof(float),cudaMemcpyHostToDevice));
 if(mode){RC(cudaMemcpy(da,a,NV*sizeof(float),cudaMemcpyHostToDevice));RC(cudaMemcpy(dl,lam,NV*sizeof(float),cudaMemcpyHostToDevice));}
 if(mode)row_vjp<<<(nr+31)/32,32>>>(nr,di,da,dl,do_);else soft<<<(nr+31)/32,32>>>(nr,h,di,do_);
 RC(cudaGetLastError());RC(cudaDeviceSynchronize());RC(cudaMemcpy(output,do_,nr*no*sizeof(float),cudaMemcpyDeviceToHost));
 cudaFree(di);cudaFree(da);cudaFree(dl);cudaFree(do_);return 0;
}
'''


@dataclass(frozen=True)
class SoftDistanceDerivative:
    impedance: np.ndarray
    D: np.ndarray
    aref: np.ndarray


@dataclass(frozen=True)
class ConstraintDataVJP:
    jacobian: np.ndarray
    D: np.ndarray
    effective_aref: np.ndarray


class CudaConstraintAdjoint:
    """Psi=lambda.T J.T phi'(Ja-aref)。返回其行输入偏导，而非完整step梯度。

    固定a/lambda、行集合/分支及frictionloss上界；每实例串行使用。
    distance_derivative保留原kbimp_d/efc_D_d，固定材料、速度和基座投影。
    """
    def __init__(self,nv,*,numerics='mixed64'):
        if not isinstance(nv,(int,np.integer)) or isinstance(nv,(bool,np.bool_)) or nv<1:raise ValueError('nv必须是正整数')
        self.nv=int(nv);self.numeric=Numerics(numerics)
        with tempfile.TemporaryDirectory(prefix='nabla_constraint_adj_') as tmp:
            cu,so=Path(tmp)/'adj.cu',Path(tmp)/'adj.so';cu.write_text(self.numeric.cuda(f'#define NV {nv}\n'+CUDA))
            compile_cuda(cu,so,extra=('-shared','-Xcompiler','-fPIC'));self._library=ctypes.CDLL(str(so))
        self._run=self._library.run;p=self.numeric.pointer
        self._run.argtypes=[ctypes.c_int,ctypes.c_int,self.numeric.scalar,p,p,p,p];self._run.restype=ctypes.c_int

    def _evaluate(self,mode,packed,a,lam,h):
        packed=self.numeric.finite(packed,'约束反向输入');nr=len(packed)
        out=np.empty((nr,3 if mode==0 else self.nv+2),self.numeric.dtype)
        if nr:
            p=self.numeric.pointer
            status=self._run(mode,nr,h,packed.ctypes.data_as(p),a.ctypes.data_as(p),lam.ctypes.data_as(p),out.ctypes.data_as(p))
            if status or not np.isfinite(out).all():raise RuntimeError(f'CUDA约束反向失败:{status}')
        return out

    def distance_derivative(self,solref,solimp,distance,invweight,h):
        pos=np.asarray(distance);nr=len(pos) if pos.ndim==1 else -1
        sr=np.asarray(solref);si=np.asarray(solimp);iw=np.asarray(invweight)
        if nr<0 or sr.shape!=(nr,2) or si.shape!=(nr,5) or iw.shape!=(nr,):raise ValueError('软参数形状不一致')
        if np.any(sr<=0) or np.any(iw<0) or not np.isfinite(h) or h<=0:raise ValueError('需要正solref/步长、非负invweight')
        z=np.zeros(self.nv,self.numeric.dtype)
        out=self._evaluate(0,np.column_stack([sr,si,pos,iw]),z,z,h)
        out.setflags(write=False);return SoftDistanceDerivative(out[:,0],out[:,1],out[:,2])

    def row_vjp(self,table,evaluation,acceleration,lam):
        nr=len(table.D);a=self.numeric.finite(acceleration,'acceleration');l=self.numeric.finite(lam,'lambda')
        if a.shape!=(self.nv,) or l.shape!=(self.nv,) or table.jacobian.shape!=(nr,self.nv):raise ValueError('约束反向维度不一致')
        for value in (evaluation.residual,evaluation.row_force,evaluation.row_curvature,evaluation.zone):
            if np.shape(value)!=(nr,):raise ValueError('前向行记录形状不一致')
        if not np.isin(evaluation.zone,[0,1,2,3]).all():raise ValueError('非法约束分支')
        out=self._evaluate(1,np.c_[table.jacobian,evaluation.residual,evaluation.row_force,evaluation.row_curvature,evaluation.zone],a,l,1.)
        out.setflags(write=False);return ConstraintDataVJP(out[:,:self.nv],out[:,self.nv],out[:,self.nv+1])
