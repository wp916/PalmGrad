"""CUDA状态积分：由implicitfast求解器提供积分加速度。

积分复用原step.STEP_DEVICE的integrate_root，对每个物体独立调用。
手掌不被本积分器推进：调用方下一步必须提供新时刻的PalmMotion。
"""

import ctypes
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np

from nablaengine.cuda import compile_cuda
from nablaengine.g1_engine import source, step as g1_step
from nablaengine.model.hand import HandState
from .kinematics import _finite32
from .numerics import Numerics


def _original_root_integrator():
    """提取原函数原文；若上游签名改变则明确失败，不另写一个同名近似实现。"""
    text = g1_step.STEP_DEVICE
    signature = "__device__ __noinline__ void integrate_root("
    start = text.index(signature)
    brace = text.index("{", start)
    depth = 1
    end = brace + 1
    while depth:
        if text[end] == "{":
            depth += 1
        elif text[end] == "}":
            depth -= 1
        end += 1
    return text[start:end]


CUDA = r"""
__device__ __constant__ int hs_kind[NB];
__global__ void hs_integrate(int nw,float h,const float* states,const float* acceleration,float* output){
  int w=blockIdx.x;if(w>=nw)return;
  const float* state=states+w*(NQ+NV);const float* a=acceleration+w*NV;
  float* q=output+w*(NQ+NV);float* v=q+NQ;
  for(int i=0;i<NQ;i++)q[i]=state[i];
  // 先用旧时刻加速度更新速度，再用新速度更新位置/姿态。
  for(int i=0;i<NV;i++)v[i]=state[NQ+i]+h*a[i];
  for(int b=1;b<NB;b++){
    if(hs_kind[b]==2)q[c_qadr[b]]+=h*v[c_dadr[b]];
    else if(hs_kind[b]==3)integrate_root(q+c_qadr[b],v+c_dadr[b],h);
  }
}
extern "C" int run(int nw,float h,const float* state,const float* a,const int* meta,float* output){
  cudaError_t e;float *ds=nullptr,*da=nullptr,*dout=nullptr;
  #define TCHECK(call) if((e=(call))!=cudaSuccess){cudaFree(ds);cudaFree(da);cudaFree(dout);return(int)e;}
  TCHECK(cudaMemcpyToSymbol(c_qadr,meta+NB,NB*sizeof(int)));
  TCHECK(cudaMemcpyToSymbol(c_dadr,meta+2*NB,NB*sizeof(int)));
  TCHECK(cudaMemcpyToSymbol(hs_kind,meta+3*NB,NB*sizeof(int)));
  TCHECK(cudaMalloc(&ds,(size_t)nw*(NQ+NV)*sizeof(float)));
  TCHECK(cudaMalloc(&da,(size_t)nw*NV*sizeof(float)));
  TCHECK(cudaMalloc(&dout,(size_t)nw*(NQ+NV)*sizeof(float)));
  TCHECK(cudaMemcpy(ds,state,(size_t)nw*(NQ+NV)*sizeof(float),cudaMemcpyHostToDevice));
  TCHECK(cudaMemcpy(da,a,(size_t)nw*NV*sizeof(float),cudaMemcpyHostToDevice));
  hs_integrate<<<nw,1>>>(nw,h,ds,da,dout);TCHECK(cudaGetLastError());TCHECK(cudaDeviceSynchronize());
  TCHECK(cudaMemcpy(output,dout,(size_t)nw*(NQ+NV)*sizeof(float),cudaMemcpyDeviceToHost));
  cudaFree(ds);cudaFree(da);cudaFree(dout);return 0;
}
"""


class CudaStateIntegrator:
    """共用CUDA状态推进算子：接收已经选定的积分加速度，不负责选择动力学算法。

    复用原integrate_root；implicitfast完成动力学求解后调用本算子。
    """

    def __init__(self, dynamics, *, numerics="float32"):
        self.numeric = Numerics(numerics)
        self.dynamics = dynamics
        self.layout = dynamics.layout
        model = SimpleNamespace(nbody=dynamics.motion.nb, nq=self.layout.nq, nv=self.layout.nv)
        code = source.header(model, 1, 1) + source.DEVICE + _original_root_integrator() + CUDA
        with tempfile.TemporaryDirectory(prefix="nabla_hand_step_") as tmp:
            cu, so = Path(tmp) / "integrate.cu", Path(tmp) / "integrate.so"
            cu.write_text(self.numeric.cuda(code))
            compile_cuda(cu, so, extra=("-shared", "-Xcompiler", "-fPIC"))
            self._library = ctypes.CDLL(str(so))
        self._run = self._library.run
        fp = self.numeric.pointer
        ip = ctypes.POINTER(ctypes.c_int)
        self._run.argtypes = [ctypes.c_int, self.numeric.scalar, fp, fp, ip, fp]
        self._run.restype = ctypes.c_int

    @staticmethod
    def _dt(dt):
        value = float(dt)
        if not np.isfinite(value) or value <= 0:
            raise ValueError("dt必须是有限正数，单位秒")
        rounded = _finite32(value, "dt")
        if rounded == 0:
            raise ValueError("dt小于float32可表示的正数范围")
        return value

    def integrate(self, state, qacc, dt):
        """给定加速度，独立验证积分；不调用动力学，不积分手掌。"""
        dt = self._dt(dt)
        self.layout.validate_state(state)
        acceleration = self.numeric.finite(qacc, "qacc")
        if acceleration.shape != (self.layout.nv,):
            raise ValueError("qacc长度必须等于动态nv")
        packed = self.numeric.finite(np.r_[state.qpos, state.qvel], "state")
        out = np.empty(self.layout.nq + self.layout.nv, self.numeric.dtype)
        fp = self.numeric.pointer
        ip = ctypes.POINTER(ctypes.c_int)
        status = self._run(
            1,
            dt,
            packed.ctypes.data_as(fp),
            acceleration.ctypes.data_as(fp),
            self.dynamics.motion.meta.ctypes.data_as(ip),
            out.ctypes.data_as(fp),
        )
        if status:
            raise RuntimeError(f"CUDA积分失败，错误码{status}")
        if not np.isfinite(out).all():
            raise RuntimeError("CUDA积分产生非有限输出")
        result = HandState(out[: self.layout.nq], out[self.layout.nq :])
        self.layout.validate_state(result)
        return result
