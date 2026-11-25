"""法向软接触行的CUDA参数装配；不检测碰撞，不求接触力。

直接提取原G1设备函数，不改写kbimp/efc_D数学。调用方传入同一时刻
的几何、材料、完整接触速度，以及规定手掌加速度的投影。
"""
import ctypes
from dataclasses import dataclass
from pathlib import Path
import tempfile
import numpy as np
from nablaengine.cuda import compile_cuda
from nablaengine.g1_engine.step import STEP_DEVICE
from .numerics import Numerics


def original_function(name):
    """提取完整设备函数；源函数缺失或格式改变时立即失败。"""
    marker='__device__ __forceinline__ '
    at=STEP_DEVICE.index(name+'(')
    start=STEP_DEVICE.rfind(marker,0,at)
    if start<0:raise RuntimeError(f'找不到原设备函数{name}')
    brace=STEP_DEVICE.index('{',at);depth=1;end=brace+1
    while depth:
        depth+=(STEP_DEVICE[end]=='{')-(STEP_DEVICE[end]=='}');end+=1
    return STEP_DEVICE[start:end]


@dataclass(frozen=True)
class NormalSoftRows:
    stiffness: np.ndarray
    damping: np.ndarray
    impedance: np.ndarray
    D: np.ndarray
    aref: np.ndarray
    effective_aref: np.ndarray


CUDA=r'''
#include <cuda_runtime.h>
#include <math.h>
#define SOLIMP_POW2 0
#define RMUL(a,b) __fmul_rn((a),(b))
#define RADD(a,b) __fadd_rn((a),(b))
#define RSUB(a,b) __fsub_rn((a),(b))
#define RFMA(a,b,c) __fmaf_rn((a),(b),(c))
'''+ '\n'.join(original_function(n) for n in ('pow_imp','imp_raw','kbimp','efc_D'))+r'''
__global__ void normal_rows(int n,float h,const float* input,float* output){
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
  // 每行：solref[2],solimp[5],distance,invweight,完整速度,Jp*ap。
  const float* x=input+11*i;float* y=output+6*i;
  kbimp(x,x+2,x[7],h,y,y+1,y+2);
  y[3]=efc_D(x[8],y[2]);
  y[4]=-y[0]*y[2]*x[7]-y[1]*x[9];
  // 求解未知量仅为动态加速度：Jd*ad-aref_eff = Jd*ad+Jp*ap-aref。
  y[5]=y[4]-x[10];
}
extern "C" int run(int n,float h,const float* in,float* out){
  float *a=nullptr,*b=nullptr;cudaError_t e;
  #define CHECK(call) if((e=(call))!=cudaSuccess){cudaFree(a);cudaFree(b);return(int)e;}
  CHECK(cudaMalloc(&a,n*11*sizeof(float)));CHECK(cudaMalloc(&b,n*6*sizeof(float)));
  CHECK(cudaMemcpy(a,in,n*11*sizeof(float),cudaMemcpyHostToDevice));
  normal_rows<<<(n+31)/32,32>>>(n,h,a,b);CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());
  CHECK(cudaMemcpy(out,b,n*6*sizeof(float),cudaMemcpyDeviceToHost));
  cudaFree(a);cudaFree(b);return 0;
}
'''


class CudaNormalSoftRows:
    """仅condim=1、margin=gap=0、正solref、refsafe开启的接触行。

    invweight是源模型双方body_invweight0平移项之和，不是J M^-1 J^T。
    本入口不接受condim=3的摩擦行；其invweight缩放与行数另行适配。
    """
    def __init__(self,plan,*,numerics="float32"):
        self.numeric=Numerics(numerics)
        self.plan=plan
        with tempfile.TemporaryDirectory(prefix='nabla_hand_rows_') as tmp:
            cu,so=Path(tmp)/'rows.cu',Path(tmp)/'rows.so'
            cu.write_text(self.numeric.cuda(CUDA));compile_cuda(cu,so,extra=('-shared','-Xcompiler','-fPIC'))
            self._library=ctypes.CDLL(str(so))
        self._run=self._library.run;fp=self.numeric.pointer
        self._run.argtypes=[ctypes.c_int,self.numeric.scalar,fp,fp];self._run.restype=ctypes.c_int

    def assemble(self,solref,solimp,distance,invweight,velocity,base_acceleration):
        distance=np.asarray(distance,dtype=float)
        if distance.ndim!=1:raise ValueError('distance必须是一维数组')
        n=len(distance);self.plan.check_counts(n,n)
        arrays=[np.asarray(a,dtype=float) for a in (solref,solimp,invweight,velocity,base_acceleration)]
        expected=[(n,2),(n,5),(n,),(n,),(n,)]
        if any(a.shape!=s for a,s in zip(arrays,expected)):raise ValueError('软接触输入形状不一致')
        sr,si,iw,v,ap=arrays
        if not all(np.isfinite(a).all() for a in [distance,*arrays]):raise ValueError('输入必须有限')
        if np.any(sr<=0):raise ValueError('当前只支持正solref；禁止把直接刚度格式套入时间常数公式')
        if np.any(distance>=0):raise ValueError('当前只接受活动的负间距、零margin法向接触')
        if np.any(iw<0):raise ValueError('invweight不能为负')
        packed=self.numeric.finite(np.column_stack([sr,si,distance,iw,v,ap]),'软接触行')
        out=np.empty((n,6),self.numeric.dtype)
        if n:
            fp=self.numeric.pointer
            status=self._run(n,self.plan.physics['timestep'],packed.ctypes.data_as(fp),out.ctypes.data_as(fp))
            if status:raise RuntimeError(f'CUDA软接触行失败，错误码{status}')
        if not np.isfinite(out).all():raise RuntimeError('CUDA软接触行产生非有限输出')
        columns=[out[:,i].copy() for i in range(6)]
        for a in columns:a.setflags(write=False)
        return NormalSoftRows(*columns)
