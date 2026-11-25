"""接触行对位置/姿态扰动的解析 CUDA 方向导数（JVP）。

evaluate输入状态、固定手掌、接触几何及其位置/接触架扰动，输出J、Jp和速度投影的同方向变化。
include_torsion=True额外计算绕法向角运动行；法向变化与刚体旋转轴变化都参与求导。
不使用数值差分，也不冻结几何冒充完整位置梯度。几何扰动由调用方提供，完整单步反向由state_adjoint组合。
当前验证精度模式为mixed64；输出不包含材料或规定手掌自身的梯度。
"""

import ctypes, tempfile
from pathlib import Path
from types import SimpleNamespace
from dataclasses import dataclass
import numpy as np
from nablaengine.cuda import compile_cuda
from nablaengine.g1_engine import source
from nablaengine.model.hand import PalmMotion
from .contacts import ContactFrame
from .contact_precision import CUDA as POSES
from .numerics import Numerics

CUDA = (
    r"""
__device__ __constant__ int jc_kind[NB];
"""
    + POSES
    + r"""
__global__ void pose_tangents(const double* exact,const double* poses,const double* direction,double* delta){
 const double* axes=exact+NQ+7+7*NB;
 for(int k=0;k<NB*6;k++)delta[k]=0;
 for(int b=1;b<NB;b++){
  int p=c_parent[b],kind=jc_kind[b],vi=c_dadr[b];double* dst=delta+6*b;
  const double* parent=delta+6*p;const double* x=poses+7*b;const double* xp=poses+7*p;
  if(kind==2){
   double r[3],dx[3],axis[3];for(int k=0;k<3;k++)r[k]=x[k]-xp[k];
   cp_cross(parent+3,r,dx);cp_qrot(x+3,axes+3*b,axis);
   for(int k=0;k<3;k++){dst[k]=parent[k]+dx[k];dst[3+k]=parent[3+k]+axis[k]*direction[vi];}
  }else if(kind==3){for(int k=0;k<3;k++)dst[k]=direction[vi+k];cp_qrot(x+3,direction+vi+3,dst+3);}
 }
}
__global__ void rows(int nc,int ndir,const double* geom,const int* bodies,const double* exact,const double* poses,
                     const double* delta,const double* velocity,double* output){
 int index=blockIdx.x*blockDim.x+threadIdx.x;if(index>=nc*ndir)return;
 int c=index/ndir,axisindex=index%ndir;bool rotation_row=axisindex==3;if(rotation_row)axisindex=0;const double* g=geom+24*c;
 const double* pos=g;const double* n=g+3+3*axisindex;const double* dpos=g+12;const double* dn=g+15+3*axisindex;
 double* row=output+index*(NV+9);for(int j=0;j<NV+9;j++)row[j]=0;
 const double* axes=exact+NQ+7+7*NB;
 for(int side=0;side<2;side++){
  double sign=side?1:-1;
  for(int b=bodies[2*c+side];b>0;b=c_parent[b]){
   int kind=jc_kind[b],vi=c_dadr[b];const double* pose=poses+7*b;const double* dp=delta+6*b;
   double r[3],dr[3],ang[3],dang[3],u[3];
   for(int k=0;k<3;k++){r[k]=pos[k]-pose[k];dr[k]=dpos[k]-dp[k];}
   cp_cross(r,n,ang);cp_cross(dr,n,dang);cp_cross(r,dn,u);for(int k=0;k<3;k++)dang[k]+=u[k];
   if(rotation_row)for(int k=0;k<3;k++){ang[k]=n[k];dang[k]=dn[k];}
   if(kind==2){double z[3],dz[3];cp_qrot(pose+3,axes+3*b,z);cp_cross(dp+3,z,dz);
    row[vi]+=sign*(cp_dot(dz,ang)+cp_dot(z,dang));
   }else if(kind==3){
    for(int k=0;k<3;k++){double unit[3]={},z[3],dz[3];unit[k]=1;cp_qrot(pose+3,unit,z);cp_cross(dp+3,z,dz);
     if(!rotation_row)row[vi+k]+=sign*dn[k];row[vi+3+k]+=sign*(cp_dot(dz,ang)+cp_dot(z,dang));}
   }else if(kind==1){for(int k=0;k<3;k++){if(!rotation_row)row[NV+k]+=sign*dn[k];row[NV+3+k]+=sign*dang[k];}}
  }
 }
 double knownv=0,knowna=0,total=0;
 for(int j=0;j<NV;j++)total+=row[j]*velocity[j];
 for(int j=0;j<6;j++){knownv+=row[NV+j]*velocity[NV+j];knowna+=row[NV+j]*velocity[NV+6+j];}
 row[NV+6]=total+knownv;row[NV+7]=knownv;row[NV+8]=knowna;
}
extern "C" int run(int nc,int ndir,const double* exact,const int* meta,const double* direction,const double* geom,
                    const int* bodies,const double* velocity,double* output){
 double *de=nullptr,*dd=nullptr,*dg=nullptr,*dv=nullptr,*dp=nullptr,*dt=nullptr,*dout=nullptr;int* db=nullptr;cudaError_t e;
 #define PC(call) if((e=(call))!=cudaSuccess){cudaFree(de);cudaFree(dd);cudaFree(dg);cudaFree(dv);cudaFree(dp);cudaFree(dt);cudaFree(dout);cudaFree(db);return(int)e;}
 PC(cudaMemcpyToSymbol(c_parent,meta,NB*sizeof(int)));PC(cudaMemcpyToSymbol(c_qadr,meta+NB,NB*sizeof(int)));
 PC(cudaMemcpyToSymbol(c_dadr,meta+2*NB,NB*sizeof(int)));PC(cudaMemcpyToSymbol(jc_kind,meta+3*NB,NB*sizeof(int)));
 PC(cudaMalloc(&de,(NQ+7+10*NB)*sizeof(double)));PC(cudaMalloc(&dd,NV*sizeof(double)));
 PC(cudaMalloc(&dg,nc*24*sizeof(double)));PC(cudaMalloc(&db,nc*2*sizeof(int)));PC(cudaMalloc(&dv,(NV+12)*sizeof(double)));
 PC(cudaMalloc(&dp,7*NB*sizeof(double)));PC(cudaMalloc(&dt,6*NB*sizeof(double)));PC(cudaMalloc(&dout,nc*ndir*(NV+9)*sizeof(double)));
 PC(cudaMemcpy(de,exact,(NQ+7+10*NB)*sizeof(double),cudaMemcpyHostToDevice));PC(cudaMemcpy(dd,direction,NV*sizeof(double),cudaMemcpyHostToDevice));
 PC(cudaMemcpy(dg,geom,nc*24*sizeof(double),cudaMemcpyHostToDevice));PC(cudaMemcpy(db,bodies,nc*2*sizeof(int),cudaMemcpyHostToDevice));
 PC(cudaMemcpy(dv,velocity,(NV+12)*sizeof(double),cudaMemcpyHostToDevice));
 cp_poses<<<1,1>>>(de,dp);pose_tangents<<<1,1>>>(de,dp,dd,dt);rows<<<(nc*ndir+31)/32,32>>>(nc,ndir,dg,db,de,dp,dt,dv,dout);
 PC(cudaGetLastError());PC(cudaDeviceSynchronize());PC(cudaMemcpy(output,dout,nc*ndir*(NV+9)*sizeof(double),cudaMemcpyDeviceToHost));
 cudaFree(de);cudaFree(dd);cudaFree(dg);cudaFree(dv);cudaFree(dp);cudaFree(dt);cudaFree(dout);cudaFree(db);return 0;
}
"""
)


@dataclass(frozen=True)
class ContactRowJVP:
    jacobian: np.ndarray
    base_jacobian: np.ndarray
    velocity: np.ndarray
    prescribed_velocity: np.ndarray
    prescribed_acceleration: np.ndarray


class CudaContactPositionJVP:
    """固定动态速度、规定手掌，给定位置/几何扰动，输出三方向或含绕法向角运动的四方向接触行JVP。

    当前只用于mixed64；与前向接触投影共用精确模型位姿。每实例串行调用。
    不自动生成几何扰动，也不提供完整step位置反向。
    """

    def __init__(self, motion):
        if motion.numeric.mode != "mixed64":
            raise ValueError("接触位置导数当前只验证mixed64")
        self.motion = motion
        self.layout = motion.layout
        self.numeric = Numerics("mixed64")
        m = SimpleNamespace(nbody=motion.nb, nq=self.layout.nq, nv=self.layout.nv)
        code = self.numeric.cuda(source.header(m, 1, 1) + source.DEVICE) + CUDA
        with tempfile.TemporaryDirectory(prefix="nabla_contact_position_") as tmp:
            cu, so = Path(tmp) / "position.cu", Path(tmp) / "position.so"
            cu.write_text(code)
            compile_cuda(cu, so, extra=("-shared", "-Xcompiler", "-fPIC"))
            self._library = ctypes.CDLL(str(so))
        self._run = self._library.run
        dp = self.numeric.pointer
        ip = ctypes.POINTER(ctypes.c_int)
        self._run.argtypes = [ctypes.c_int, ctypes.c_int, dp, ip, dp, dp, ip, dp, dp]
        self._run.restype = ctypes.c_int

    def evaluate(
        self,
        state,
        palm,
        contacts,
        direction,
        contact_position_delta,
        contact_frame_delta,
        *,
        include_torsion=False,
    ):
        self.layout.validate_state(state)
        if not isinstance(palm, PalmMotion):
            raise TypeError("需要显式PalmMotion")
        contacts = list(contacts)
        nc = len(contacts)
        nv = self.layout.nv
        for c in contacts:
            if not isinstance(c, ContactFrame):
                raise TypeError("需要ContactFrame序列")
            if c.body1 >= self.motion.nb or c.body2 >= self.motion.nb:
                raise ValueError("接触body id超出范围")
        values = []
        for value, shape, name in (
            (direction, (nv,), "direction"),
            (contact_position_delta, (nc, 3), "contact_position_delta"),
            (contact_frame_delta, (nc, 3, 3), "contact_frame_delta"),
        ):
            value = self.numeric.finite(value, name)
            if value.shape != shape:
                raise ValueError(f"{name}形状必须为{shape}")
            values.append(value)
        d, dpos, dframe = values
        for c, df in zip(contacts, dframe):
            if not np.allclose(df @ c.frame.T + c.frame @ df.T, 0, rtol=0, atol=1e-8):
                raise ValueError("frame_delta不是正交接触架的合法一阶扰动")
        direction_count = 4 if include_torsion else 3
        out = np.empty((nc, direction_count, nv + 9), np.float64)
        if nc:
            exact = self.numeric.finite(
                np.r_[state.qpos, palm.position, palm.quaternion, self.motion.geometry64],
                "精确位姿输入",
            )
            geom = self.numeric.finite(
                [
                    np.r_[c.position, c.frame.ravel(), dp, df.ravel()]
                    for c, dp, df in zip(contacts, dpos, dframe)
                ],
                "接触几何扰动",
            )
            bodies = np.ascontiguousarray([[c.body1, c.body2] for c in contacts], dtype=np.int32)
            v = self.numeric.finite(
                np.r_[
                    state.qvel,
                    palm.linear_velocity,
                    palm.angular_velocity,
                    palm.linear_acceleration,
                    palm.angular_acceleration,
                ],
                "速度/规定加速度",
            )
            dp = self.numeric.pointer
            ip = ctypes.POINTER(ctypes.c_int)
            status = self._run(
                nc,
                direction_count,
                exact.ctypes.data_as(dp),
                self.motion.meta.ctypes.data_as(ip),
                d.ctypes.data_as(dp),
                geom.ctypes.data_as(dp),
                bodies.ctypes.data_as(ip),
                v.ctypes.data_as(dp),
                out.ctypes.data_as(dp),
            )
            if status or not np.isfinite(out).all():
                raise RuntimeError(f"CUDA接触行位置JVP失败:{status}")
        out.setflags(write=False)
        return ContactRowJVP(
            out[:, :, :nv],
            out[:, :, nv : nv + 6],
            out[:, :, nv + 6],
            out[:, :, nv + 7],
            out[:, :, nv + 8],
        )
