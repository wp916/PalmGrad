"""规定运动基座、多自由物体的实际CUDA运动学。

hinge递推调用原引擎source.DEVICE里的fwd_hinge_body；基座和多free根初始化为新实现。
每环境一个线程顺序遍历树，作为可验证实现，不是最终性能方案。
所有输出速度/加速度均为世界系，线运动量位于刚体原点；不减去重力、不推进时间。
"""
import ctypes
from dataclasses import dataclass
from pathlib import Path
import tempfile

import numpy as np

from nablaengine.cuda import compile_cuda
from nablaengine.g1_engine import source
from nablaengine.model.hand import HandLayout, PalmMotion
from .numerics import Numerics


CUDA = r'''
#define INPUT_SIZE (NQ+2*NV+19)
#define OUTPUT_SIZE (NB*19)
__device__ __constant__ int hk_kind[NB];
__global__ void hk_forward(int nw,const float* inputs,float* outputs){
  int world=blockIdx.x;
  if(world>=nw)return;
  const float* q=inputs+world*INPUT_SIZE;
  const float* v=q+NQ;const float* a=v+NV;const float* palm=a+NV;
  float Q[NB][4]={},X[NB][3]={},W[NB][3]={},V[NB][3]={},AL[NB][3]={},A[NB][3]={};
  Q[0][0]=1.f;
  for(int b=1;b<NB;b++){
    int kind=hk_kind[b];
    if(kind==1){
      for(int k=0;k<3;k++){
        X[b][k]=palm[k];W[b][k]=palm[7+k];V[b][k]=palm[10+k];
        AL[b][k]=palm[13+k];A[b][k]=palm[16+k];
      }
      for(int k=0;k<4;k++)Q[b][k]=palm[3+k];
    }else if(kind==2){
      int p=c_parent[b];
      fwd_hinge_body(b,Q[p],X[p],W[p],V[p],AL[p],A[p],q,v,a,
                     Q[b],X[b],W[b],V[b],AL[b],A[b]);
    }else if(kind==3){
      int qi=c_qadr[b],vi=c_dadr[b];
      for(int k=0;k<3;k++){X[b][k]=q[qi+k];V[b][k]=v[vi+k];A[b][k]=a[vi+k];}
      for(int k=0;k<4;k++)Q[b][k]=q[qi+3+k];
      // free速度中的角速度/角加速度为物体系，输出统一转为世界系。
      qrot_rn(Q[b],v+vi+3,W[b]);qrot_rn(Q[b],a+vi+3,AL[b]);
    }
  }
  for(int b=0;b<NB;b++){
    float* dst=outputs+world*OUTPUT_SIZE+b*19;
    for(int k=0;k<3;k++){
      dst[k]=X[b][k];dst[7+k]=W[b][k];dst[10+k]=V[b][k];
      dst[13+k]=AL[b][k];dst[16+k]=A[b][k];
    }
    for(int k=0;k<4;k++)dst[3+k]=Q[b][k];
  }
}
extern "C" int run(int nw,const float* inputs,const int* meta,const float* fixed,float* outputs){
  cudaError_t e;float *di=nullptr,*dout=nullptr;
  #define CHECK(call) if((e=(call))!=cudaSuccess){cudaFree(di);cudaFree(dout);return(int)e;}
  CHECK(cudaMemcpyToSymbol(c_parent,meta,NB*sizeof(int)));
  CHECK(cudaMemcpyToSymbol(c_qadr,meta+NB,NB*sizeof(int)));
  CHECK(cudaMemcpyToSymbol(c_dadr,meta+2*NB,NB*sizeof(int)));
  CHECK(cudaMemcpyToSymbol(hk_kind,meta+3*NB,NB*sizeof(int)));
  CHECK(cudaMemcpyToSymbol(c_pos,fixed,3*NB*sizeof(float)));
  CHECK(cudaMemcpyToSymbol(c_quat,fixed+3*NB,4*NB*sizeof(float)));
  CHECK(cudaMemcpyToSymbol(c_axis,fixed+7*NB,3*NB*sizeof(float)));
  CHECK(cudaMalloc(&di,(size_t)nw*INPUT_SIZE*sizeof(float)));
  CHECK(cudaMalloc(&dout,(size_t)nw*OUTPUT_SIZE*sizeof(float)));
  CHECK(cudaMemcpy(di,inputs,(size_t)nw*INPUT_SIZE*sizeof(float),cudaMemcpyHostToDevice));
  hk_forward<<<nw,1>>>(nw,di,dout);
  CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());
  CHECK(cudaMemcpy(outputs,dout,(size_t)nw*OUTPUT_SIZE*sizeof(float),cudaMemcpyDeviceToHost));
  cudaFree(di);cudaFree(dout);return 0;
}
'''


@dataclass(frozen=True)
class BodyMotion:
    """每行对应源模型body id，含world。数组互不依赖输入，初始标为只读。"""
    position: np.ndarray
    quaternion: np.ndarray
    angular_velocity: np.ndarray
    linear_velocity: np.ndarray
    angular_acceleration: np.ndarray
    linear_acceleration: np.ndarray


def _finite32(values, name):
    with np.errstate(over='ignore',invalid='ignore'):
        result=np.ascontiguousarray(values,dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError(f'{name} 超出float32有限范围')
    return result


class CudaKinematics:
    """模型布局建立后编译一次。forward/forward_many只算运动学；不求加速度。

    元数据在每次调用时装载到CUDA常量；同一实例不支持并发调用。
    模型在构造后视为冻结，后续修改源MjModel不会修改本实例。
    """
    def __init__(self, model, layout: HandLayout, *, numerics="float32"):
        self.numeric=Numerics(numerics)
        from .layout import build_topology
        self.topology=build_topology(model,layout)
        self.layout=layout;self.nb=int(model.nbody)
        self.meta=self.topology.meta
        self.fixed=self.numeric.finite(np.concatenate([model.body_pos.ravel(),model.body_quat.ravel(),self.topology.axes.ravel()]),'模型常量')
        # 接触力臂需保留源精度，不能把已经舍入的fixed再转double冒充恢复精度。
        self.geometry64=np.ascontiguousarray(np.r_[model.body_pos.ravel(),model.body_quat.ravel(),self.topology.axes.ravel()],dtype=np.float64)
        self.geometry64.setflags(write=False)
        # 只使用原引擎header/DEVICE，不调用G1 constants()或其完整前向kernel。
        code=source.header(model,int(self.topology.depth.max()),1)+source.DEVICE+CUDA
        with tempfile.TemporaryDirectory(prefix='nabla_hand_kinematics_') as tmp:
            cu,so=Path(tmp)/'kinematics.cu',Path(tmp)/'kinematics.so'
            cu.write_text(self.numeric.cuda(code));compile_cuda(cu,so,extra=('-shared','-Xcompiler','-fPIC'))
            self._library=ctypes.CDLL(str(so))
        self._run=self._library.run
        fp=self.numeric.pointer;ip=ctypes.POINTER(ctypes.c_int)
        self._run.argtypes=[ctypes.c_int,fp,ip,fp,fp];self._run.restype=ctypes.c_int

    def forward(self,state,palm: PalmMotion,qacc):
        return self.forward_many([(state,palm,qacc)])[0]

    def forward_many(self,cases):
        """批量独立工况，每项为(state,palm,指定qacc)；不是时间轨迹积分。"""
        cases=list(cases)
        if not cases:raise ValueError('至少需要一个工况')
        packed=[]
        for state,palm,qacc in cases:
            self.layout.validate_state(state)
            if not isinstance(palm,PalmMotion):raise TypeError('需要显式PalmMotion')
            a=np.asarray(qacc,dtype=np.float64)
            if a.shape!=(self.layout.nv,) or not np.isfinite(a).all():
                raise ValueError('qacc必须是完整动态自由度长度的有限加速度数组')
            packed.append(np.concatenate([state.qpos,state.qvel,a,palm.position,palm.quaternion,
                                          palm.angular_velocity,palm.linear_velocity,
                                          palm.angular_acceleration,palm.linear_acceleration]))
        inputs=self.numeric.finite(packed,'运动学输入')
        out=np.empty((len(cases),self.nb,19),self.numeric.dtype)
        fp=self.numeric.pointer;ip=ctypes.POINTER(ctypes.c_int)
        status=self._run(len(cases),inputs.ctypes.data_as(fp),self.meta.ctypes.data_as(ip),
                         self.fixed.ctypes.data_as(fp),out.ctypes.data_as(fp))
        if status:raise RuntimeError(f'CUDA运动学失败，错误码{status}')
        if not np.isfinite(out).all():raise RuntimeError('CUDA运动学产生非有限输出')
        result=[]
        for row in out:
            arrays=[row[:,s].copy() for s in [slice(0,3),slice(3,7),slice(7,10),slice(10,13),slice(13,16),slice(16,19)]]
            for a in arrays:a.setflags(write=False)
            result.append(BodyMotion(*arrays))
        return result
