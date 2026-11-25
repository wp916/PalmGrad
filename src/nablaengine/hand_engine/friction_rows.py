"""将接触方向装配为原金字塔摩擦行，支持滑动摩擦和绕法向的扭转摩擦。

CudaPyramidRows.assemble输入接触Jacobian/速度、软参数、距离、invweight和滑动摩擦系数。
省略torsion_friction时输出每接触4行；指定正扭转系数（单位m）时需要4方向输入，输出6行。
返回每行J/Jp、D、参考加速度等，供Newton联合求解；本文件不独立决定接触力。
沿用原kbimp/efc_D及MuJoCo金字塔common R规则，不能给扭转边另加任意角速度阻尼。
"""

import ctypes
from dataclasses import dataclass
from pathlib import Path
import tempfile
import numpy as np
from nablaengine.cuda import compile_cuda
from .constraint_rows import CUDA as SOFT_CUDA, original_function
from .contacts import FrameRows
from .numerics import Numerics


def _invweight_function():
    # 仅改寻址/实参：原pr查表改成显式输入；沿用原common R缩放：依赖首个滑动mu，扭转两边与其共用D。
    code = original_function("contact_invw")
    changes = {
        "contact_invw(int pr, float mu)": "hand_contact_invw(float raw_invweight, float mu, float impratio)",
        "RADD(c_binvw[c_gbody[PG1(pr)]], c_binvw[c_gbody[PG2(pr)]])": "raw_invweight",
        "if (PDIM(pr) == 3)": "",
        "c_sopt[0]": "impratio",
    }
    for old, new in changes.items():
        if code.count(old) != 1:
            raise RuntimeError("原contact_invw发生变化，需重新核对适配")
        code = code.replace(old, new)
    return code


CUDA = SOFT_CUDA.split("__global__ void normal_rows")[0] + _invweight_function() + r"""
#define STRIDE (NV+15)
__global__ void pyramid_rows(int nc,int edges,float h,float impratio,const float* material,const float* frames,float* out){
  int r=blockIdx.x*blockDim.x+threadIdx.x;if(r>=edges*nc)return;
  int c=r/edges,t=r%edges,axis=1+(t>>1);float sign=(t&1)?-1.f:1.f;
  const float* x=material+11*c;float mu=x[9];float weight=axis==3?x[10]:mu;
  const float* n=frames+c*(edges/2+1)*(NV+9);const float* tangent=n+axis*(NV+9);
  float* y=out+r*STRIDE;
  for(int j=0;j<NV+9;j++)y[j]=n[j]+sign*weight*tangent[j];
  kbimp(x,x+2,x[7],h,y+NV+9,y+NV+10,y+NV+11);
  float iw=hand_contact_invw(x[8],mu,impratio);
  y[NV+12]=efc_D(iw,y[NV+11]);
  // 软弹簧部分仍用法向间距；阻尼部分使用该金字塔行的完整速度。
  y[NV+13]=-y[NV+9]*y[NV+11]*x[7]-y[NV+10]*y[NV+6];
  y[NV+14]=y[NV+13]-y[NV+8];
}
extern "C" int run(int nc,int edges,float h,float impratio,const float* material,const float* frames,float* output){
  float *dm=nullptr,*df=nullptr,*dout=nullptr;cudaError_t e;
  #define FCHECK(call) if((e=(call))!=cudaSuccess){cudaFree(dm);cudaFree(df);cudaFree(dout);return(int)e;}
  FCHECK(cudaMalloc(&dm,nc*11*sizeof(float)));
  FCHECK(cudaMalloc(&df,nc*(edges/2+1)*(NV+9)*sizeof(float)));
  FCHECK(cudaMalloc(&dout,nc*edges*STRIDE*sizeof(float)));
  FCHECK(cudaMemcpy(dm,material,nc*11*sizeof(float),cudaMemcpyHostToDevice));
  FCHECK(cudaMemcpy(df,frames,nc*(edges/2+1)*(NV+9)*sizeof(float),cudaMemcpyHostToDevice));
  pyramid_rows<<<(edges*nc+31)/32,32>>>(nc,edges,h,impratio,dm,df,dout);
  FCHECK(cudaGetLastError());FCHECK(cudaDeviceSynchronize());
  FCHECK(cudaMemcpy(output,dout,nc*edges*STRIDE*sizeof(float),cudaMemcpyDeviceToHost));
  cudaFree(dm);cudaFree(df);cudaFree(dout);return 0;
}
"""


@dataclass(frozen=True)
class PyramidRows:
    jacobian: np.ndarray
    base_jacobian: np.ndarray
    velocity: np.ndarray
    prescribed_velocity: np.ndarray
    prescribed_acceleration: np.ndarray
    stiffness: np.ndarray
    damping: np.ndarray
    impedance: np.ndarray
    D: np.ndarray
    aref: np.ndarray
    effective_aref: np.ndarray
    kind: np.ndarray
    contact_index: np.ndarray
    edge: np.ndarray


class CudaPyramidRows:
    """接受condim3/4、两切向同一正mu、零margin/gap、正solref、refsafe开启。

    行顺序n±mu*t1、n±mu*t2；condim4另加n±mu_torsion*angular_normal。
    D保留MuJoCo金字塔common R：所有边用首个滑动mu的invweight缩放。帧、材料、状态须属于同一时刻。
    低层输入未携带模型condim/margin/refsafe标志，调用方负责审核这些约定。
    """

    def __init__(self, plan, *, numerics="float32"):
        self.numeric = Numerics(numerics)
        self.plan = plan
        self.nv = plan.layout.nv
        code = f"#define NV {self.nv}\n" + CUDA
        with tempfile.TemporaryDirectory(prefix="nabla_hand_friction_") as tmp:
            cu, so = Path(tmp) / "friction.cu", Path(tmp) / "friction.so"
            cu.write_text(self.numeric.cuda(code))
            compile_cuda(cu, so, extra=("-shared", "-Xcompiler", "-fPIC"))
            self._library = ctypes.CDLL(str(so))
        self._run = self._library.run
        fp = self.numeric.pointer
        self._run.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            self.numeric.scalar,
            self.numeric.scalar,
            fp,
            fp,
            fp,
        ]
        self._run.restype = ctypes.c_int

    def assemble(
        self, frames, solref, solimp, distance, invweight, friction, *, torsion_friction=None
    ):
        if not isinstance(frames, FrameRows):
            raise TypeError("需要FrameRows")
        distance = np.asarray(distance, dtype=float)
        if distance.ndim != 1:
            raise ValueError("distance必须一维")
        nc = len(distance)
        nv = self.nv
        edge_count = 6 if torsion_friction is not None else 4
        direction_count = edge_count // 2 + 1
        self.plan.check_counts(nc, edge_count * nc)
        torsion = (
            np.zeros(nc) if torsion_friction is None else np.asarray(torsion_friction, dtype=float)
        )
        if (
            torsion.shape != (nc,)
            or not np.isfinite(torsion).all()
            or (torsion_friction is not None and np.any(torsion <= 0))
        ):
            raise ValueError("condim4需要每接触一个有限正扭转摩擦系数，单位m")
        inputs = [np.asarray(a, dtype=float) for a in (solref, solimp, invweight, friction)]
        sr, si, iw, mu = inputs
        if any(a.shape != s for a, s in zip(inputs, [(nc, 2), (nc, 5), (nc,), (nc,)])):
            raise ValueError("接触材料形状错误")
        arrays = [
            frames.jacobian,
            frames.base_jacobian,
            frames.velocity,
            frames.prescribed_velocity,
            frames.prescribed_acceleration,
        ]
        shapes = [(nc, direction_count, nv), (nc, direction_count, 6)] + [(nc, direction_count)] * 3
        if any(a.shape != s for a, s in zip(arrays, shapes)):
            raise ValueError("接触架行形状错误")
        if not all(np.isfinite(a).all() for a in [distance, *inputs, *arrays]):
            raise ValueError("输入必须有限")
        if np.any(sr <= 0) or np.any(mu <= 0) or np.any(iw < 0) or np.any(distance >= 0):
            raise ValueError("需要正solref、正mu、非负invweight和负间距")
        material = self.numeric.finite(
            np.column_stack([sr, si, distance, iw, mu, torsion]), "摩擦材料"
        )
        packed = self.numeric.finite(
            np.concatenate([*arrays[:2], *[a[..., None] for a in arrays[2:]]], axis=2), "接触架行"
        )
        output = np.empty((edge_count * nc, nv + 15), self.numeric.dtype)
        if nc:
            fp = self.numeric.pointer
            status = self._run(
                nc,
                edge_count,
                self.plan.physics["timestep"],
                self.plan.physics["impratio"],
                material.ctypes.data_as(fp),
                packed.ctypes.data_as(fp),
                output.ctypes.data_as(fp),
            )
            if status:
                raise RuntimeError(f"CUDA摩擦行失败，错误码{status}")
        if not np.isfinite(output).all():
            raise RuntimeError("CUDA摩擦行产生非有限输出")
        parts = [output[:, :nv].copy(), output[:, nv : nv + 6].copy()] + [
            output[:, nv + i].copy() for i in range(6, 15)
        ]
        parts += [
            np.full(edge_count * nc, 2, np.uint8),
            np.repeat(np.arange(nc, dtype=np.uint16), edge_count),
            np.tile(np.arange(edge_count, dtype=np.int8), nc),
        ]
        for a in parts:
            a.setflags(write=False)
        return PyramidRows(*parts)
