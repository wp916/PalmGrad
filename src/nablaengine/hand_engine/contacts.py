"""把接触点和接触架转换为手系统的速度/施力 Jacobian；不检测碰撞、不求力。

输入：HandState、规定手掌PalmMotion，以及当前时刻的接触几何，法向由body1指向body2。
接口：assemble生成法向行；assemble_frames生成法向/两切向，include_torsion=True再加绕法向角运动。
输出：动态列Jd、规定手掌列Jp、相对速度及规定加速度投影。第4方向的单位是角速度。
Jd转置映射接触力/矩为动态广义力；Jp用于保留规定运动贡献，自接触的公共手掌运动会抵消。
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
from .contact_precision import CUDA as PRECISION_CUDA


@dataclass(frozen=True)
class NormalContact:
    body1: int
    body2: int
    position: np.ndarray
    normal: np.ndarray
    distance: float

    def __post_init__(self):
        for name in ("body1", "body2"):
            value = getattr(self, name)
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or value < 0
            ):
                raise ValueError("body id必须是非负整数")
        if self.body1 == self.body2:
            raise ValueError("接触双方不能是同一个刚体")
        for name in ("position", "normal"):
            a = np.array(getattr(self, name), dtype=float, copy=True)
            if a.shape != (3,) or not np.isfinite(a).all():
                raise ValueError(f"{name}必须是有限世界系3向量")
            a.setflags(write=False)
            object.__setattr__(self, name, a)
        if not np.isclose(np.linalg.norm(self.normal), 1, rtol=0, atol=1e-6):
            raise ValueError("normal必须是body1指向body2的单位法向")
        distance = float(self.distance)
        if not np.isfinite(distance):
            raise ValueError("distance必须有限")
        object.__setattr__(self, "distance", distance)


@dataclass(frozen=True)
class NormalRows:
    """每个接触一行；normal_velocity=Jd*v+Jp*[v_p,omega_p]。

    prescribed_acceleration仅为Jp*[a_p,alpha_p]，不包含Jdot*v或法向变化，
    不可把它当作接触间距的完整二阶导数。point_normal_velocity是运动学直接点速度核对值。
    """

    jacobian: np.ndarray
    base_jacobian: np.ndarray
    normal_velocity: np.ndarray
    prescribed_velocity: np.ndarray
    prescribed_acceleration: np.ndarray
    point_normal_velocity: np.ndarray


@dataclass(frozen=True)
class ContactFrame:
    """世界系接触架：三行依次为normal、tangent1、tangent2，构成右手正交基。

    切向基由几何调用方提供，不在装配器中重新猜测，避免对照时混入基选择差异。
    """

    body1: int
    body2: int
    position: np.ndarray
    frame: np.ndarray
    distance: float

    def __post_init__(self):
        frame = np.array(self.frame, dtype=float, copy=True)
        if frame.shape != (3, 3) or not np.isfinite(frame).all():
            raise ValueError("frame必须是有限3x3数组")
        if not np.allclose(frame @ frame.T, np.eye(3), rtol=0, atol=1e-6) or not np.isclose(
            np.linalg.det(frame), 1, atol=1e-6
        ):
            raise ValueError("frame必须是右手正交单位接触架")
        normal = NormalContact(self.body1, self.body2, self.position, frame[0], self.distance)
        frame.setflags(write=False)
        object.__setattr__(self, "frame", frame)
        object.__setattr__(self, "position", normal.position)
        object.__setattr__(self, "distance", normal.distance)


@dataclass(frozen=True)
class FrameRows:
    """每点3或4方向：法向、两切向，以及可选的绕法向角速度。

    Jacobian 为 (ncontact,ndir,nv)，ndir=3/4；最后一方向是角速度，不是点线速度。"""

    jacobian: np.ndarray
    base_jacobian: np.ndarray
    velocity: np.ndarray
    prescribed_velocity: np.ndarray
    prescribed_acceleration: np.ndarray
    point_velocity: np.ndarray


CUDA = (
    r"""
#define J_OUTPUT (NV+10)
__device__ __constant__ int jc_kind[NB];
"""
    + PRECISION_CUDA
    + r"""
__global__ void jc_assemble(int nc,const float* frame,const double* geom,const int* body,
                            const double* poses,const double* exact,float* outputs){
  int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=nc)return;
  const float* v=frame+NB*19;const float* vp=v+NV;const float* ap=vp+6;
  const double* pos=geom+7*c;const double* n=pos+3;bool rotation_row=pos[6]!=0.;const double* axes=exact+NQ+7+7*NB;
  float* row=outputs+c*J_OUTPUT;double precise[NV+6]={};double direct=0.;
  for(int i=0;i<J_OUTPUT;i++)row[i]=0.f;
  for(int side=0;side<2;side++){
    int b=body[2*c+side];double sign=side==0?-1.:1.;
    const float* motion=frame+b*19;double r[3],omega[3],wr[3];
    for(int k=0;k<3;k++){r[k]=pos[k]-poses[7*b+k];omega[k]=motion[7+k];}
    cp_cross(omega,r,wr);
    for(int k=0;k<3;k++)direct+=sign*n[k]*(rotation_row?motion[7+k]:(motion[10+k]+wr[k]));
    for(int ancestor=b;ancestor>0;ancestor=c_parent[ancestor]){
      int kind=jc_kind[ancestor];const double* m=poses+7*ancestor;
      double lever[3],angular[3];for(int k=0;k<3;k++)lever[k]=pos[k]-m[k];
      cp_cross(lever,n,angular);
      if(rotation_row)for(int k=0;k<3;k++)angular[k]=n[k];
      if(kind==2){
        double axis[3];cp_qrot(m+3,axes+3*ancestor,axis);
        precise[c_dadr[ancestor]]+=sign*cp_dot(axis,angular);
      }else if(kind==3){
        int d=c_dadr[ancestor];
        for(int k=0;k<3;k++){
          if(!rotation_row)precise[d+k]+=sign*n[k];
          double unit[3]={0.,0.,0.},axis[3];unit[k]=1.;cp_qrot(m+3,unit,axis);
          precise[d+3+k]+=sign*cp_dot(axis,angular);
        }
      }else if(kind==1){
        for(int k=0;k<3;k++){if(!rotation_row)precise[NV+k]+=sign*n[k];precise[NV+3+k]+=sign*angular[k];}
      }
    }
  }
  // 小量力臂相减、叉积及两侧祖先消去完成后，才进入原float32求解路径。
  for(int i=0;i<NV+6;i++)row[i]=(float)precise[i];
  float known_v=0.f,known_a=0.f,total=0.f;
  for(int i=0;i<6;i++){known_v+=row[NV+i]*vp[i];known_a+=row[NV+i]*ap[i];}
  for(int i=0;i<NV;i++)total+=row[i]*v[i];
  row[NV+6]=total+known_v;row[NV+7]=known_v;row[NV+8]=known_a;row[NV+9]=(float)direct;
}
extern "C" int run(int nc,const float* frame,const double* geom,const int* bodies,const int* meta,
                   const double* exact,float* out){
  cudaError_t e;float *df=nullptr,*dout=nullptr;double *dg=nullptr,*de=nullptr,*dp=nullptr;int* db=nullptr;
  #define JCHECK(call) if((e=(call))!=cudaSuccess){cudaFree(df);cudaFree(dg);cudaFree(db);cudaFree(dout);cudaFree(de);cudaFree(dp);return(int)e;}
  JCHECK(cudaMemcpyToSymbol(c_parent,meta,NB*sizeof(int)));
  JCHECK(cudaMemcpyToSymbol(c_qadr,meta+NB,NB*sizeof(int)));
  JCHECK(cudaMemcpyToSymbol(c_dadr,meta+2*NB,NB*sizeof(int)));
  JCHECK(cudaMemcpyToSymbol(jc_kind,meta+3*NB,NB*sizeof(int)));
  JCHECK(cudaMalloc(&df,(NB*19+NV+12)*sizeof(float)));
  JCHECK(cudaMalloc(&dg,(size_t)nc*7*sizeof(double)));
  JCHECK(cudaMalloc(&de,(NQ+7+10*NB)*sizeof(double)));
  JCHECK(cudaMalloc(&dp,NB*7*sizeof(double)));
  JCHECK(cudaMalloc(&db,(size_t)nc*2*sizeof(int)));
  JCHECK(cudaMalloc(&dout,(size_t)nc*J_OUTPUT*sizeof(float)));
  JCHECK(cudaMemcpy(df,frame,(NB*19+NV+12)*sizeof(float),cudaMemcpyHostToDevice));
  JCHECK(cudaMemcpy(dg,geom,(size_t)nc*7*sizeof(double),cudaMemcpyHostToDevice));
  JCHECK(cudaMemcpy(de,exact,(NQ+7+10*NB)*sizeof(double),cudaMemcpyHostToDevice));
  JCHECK(cudaMemcpy(db,bodies,(size_t)nc*2*sizeof(int),cudaMemcpyHostToDevice));
  cp_poses<<<1,1>>>(de,dp);JCHECK(cudaGetLastError());
  jc_assemble<<<(nc+31)/32,32>>>(nc,df,dg,db,dp,de,dout);JCHECK(cudaGetLastError());JCHECK(cudaDeviceSynchronize());
  JCHECK(cudaMemcpy(out,dout,(size_t)nc*J_OUTPUT*sizeof(float),cudaMemcpyDeviceToHost));
  cudaFree(df);cudaFree(dg);cudaFree(db);cudaFree(dout);cudaFree(de);cudaFree(dp);return 0;
}
"""
)


class CudaContactJacobian:
    """motion为已建立的CudaKinematics；一个线程装配一个接触，支持显式地址/多根。

    assemble生成法向行；assemble_frames生成三方向。均不生成软参数或接触力。
    GPU双精度计算接触位姿、相对力臂与投影；行/速度精度跟随motion.numeric模式。
    不依赖MuJoCo运行时几何接口。验证脚本另外用MuJoCo提供测试几何并建立参考。
    """

    def __init__(self, motion):
        self.numeric = motion.numeric
        self.motion = motion
        self.layout = motion.layout
        self.nb = motion.nb
        model = SimpleNamespace(nbody=self.nb, nq=self.layout.nq, nv=self.layout.nv)
        code = source.header(model, 1, 1) + source.DEVICE + CUDA
        with tempfile.TemporaryDirectory(prefix="nabla_hand_contact_jac_") as tmp:
            cu, so = Path(tmp) / "contact.cu", Path(tmp) / "contact.so"
            cu.write_text(self.numeric.cuda(code))
            compile_cuda(cu, so, extra=("-shared", "-Xcompiler", "-fPIC"))
            self._library = ctypes.CDLL(str(so))
        self._run = self._library.run
        fp = self.numeric.pointer
        ip = ctypes.POINTER(ctypes.c_int)
        dp = ctypes.POINTER(ctypes.c_double)
        self._run.argtypes = [ctypes.c_int, fp, dp, ip, ip, dp, fp]
        self._run.restype = ctypes.c_int

    def assemble(self, state, palm, contacts):
        self.layout.validate_state(state)
        if not isinstance(palm, PalmMotion):
            raise TypeError("需要显式PalmMotion")
        contacts = list(contacts)
        for c in contacts:
            if not isinstance(c, NormalContact):
                raise TypeError("contacts必须是NormalContact序列")
            if c.body1 >= self.nb or c.body2 >= self.nb:
                raise ValueError("接触body id超出模型范围")
        return self._assemble_directions(state, palm, contacts)

    def assemble_frames(self, state, palm, contacts, *, include_torsion=False):
        """装配法向/两切向，可选第4方向为绕法向的相对角速度。"""
        self.layout.validate_state(state)
        if not isinstance(palm, PalmMotion):
            raise TypeError("需要显式PalmMotion")
        contacts = list(contacts)
        for c in contacts:
            if not isinstance(c, ContactFrame):
                raise TypeError("需要ContactFrame序列")
            if c.body1 >= self.nb or c.body2 >= self.nb:
                raise ValueError("接触body id超出模型范围")
        directions = []
        direction_count = 4 if include_torsion else 3
        for contact in contacts:
            for axis in contact.frame:
                directions.append(
                    SimpleNamespace(
                        body1=contact.body1,
                        body2=contact.body2,
                        position=contact.position,
                        normal=axis,
                        angular=False,
                    )
                )
            if include_torsion:
                directions.append(
                    SimpleNamespace(
                        body1=contact.body1,
                        body2=contact.body2,
                        position=contact.position,
                        normal=contact.frame[0],
                        angular=True,
                    )
                )
        flat = self._assemble_directions(state, palm, directions)
        values = []
        for value in flat.__dict__.values():
            shaped = value.reshape((len(contacts), direction_count) + value.shape[1:])
            shaped.setflags(write=False)
            values.append(shaped)
        return FrameRows(*values)

    def _assemble_directions(self, state, palm, contacts):
        # 私有入口把normal视为任意给定单位投影方向；公共入口分别验证法向/完整接触架。
        nc = len(contacts)
        nv = self.layout.nv
        if nc:
            motion = self.motion.forward(state, palm, np.zeros(nv))
            packed_motion = np.concatenate(
                [
                    motion.position,
                    motion.quaternion,
                    motion.angular_velocity,
                    motion.linear_velocity,
                    motion.angular_acceleration,
                    motion.linear_acceleration,
                ],
                axis=1,
            )
            frame = self.numeric.finite(
                np.r_[
                    packed_motion.ravel(),
                    state.qvel,
                    palm.linear_velocity,
                    palm.angular_velocity,
                    palm.linear_acceleration,
                    palm.angular_acceleration,
                ],
                "接触运动",
            )
            geom = np.ascontiguousarray(
                [
                    np.r_[c.position, c.normal, float(getattr(c, "angular", False))]
                    for c in contacts
                ],
                dtype=np.float64,
            )
            bodies = np.ascontiguousarray([[c.body1, c.body2] for c in contacts], dtype=np.int32)
            out = np.empty((nc, nv + 10), self.numeric.dtype)
            fp = self.numeric.pointer
            ip = ctypes.POINTER(ctypes.c_int)
            exact = np.ascontiguousarray(
                np.r_[state.qpos, palm.position, palm.quaternion, self.motion.geometry64],
                dtype=np.float64,
            )
            dp = ctypes.POINTER(ctypes.c_double)
            status = self._run(
                nc,
                frame.ctypes.data_as(fp),
                geom.ctypes.data_as(dp),
                bodies.ctypes.data_as(ip),
                self.motion.meta.ctypes.data_as(ip),
                exact.ctypes.data_as(dp),
                out.ctypes.data_as(fp),
            )
            if status:
                raise RuntimeError(f"CUDA接触Jacobian失败，错误码{status}")
            if not np.isfinite(out).all():
                raise RuntimeError("CUDA接触Jacobian产生非有限输出")
        else:
            out = np.empty((0, nv + 10), self.numeric.dtype)
        parts = [out[:, :nv].copy(), out[:, nv : nv + 6].copy()] + [
            out[:, nv + i].copy() for i in range(6, 10)
        ]
        for p in parts:
            p.setflags(write=False)
        return NormalRows(*parts)
