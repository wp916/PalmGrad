"""规定运动手掌的CUDA逆/前向动力学与质量矩阵列提取；尚无接触求解和step。

运动学复用CudaKinematics；惯性力、子树累积和hinge投影调用原引擎DEVICE函数。
质量矩阵由零速度、零重力下的单位加速度RNE提取，属于正确性基线，非优化CRBA。
"""
import ctypes
from dataclasses import dataclass
from pathlib import Path
import tempfile

import numpy as np

from nablaengine.cuda import compile_cuda
from nablaengine.g1_engine import source
from nablaengine.model.hand import HandState, PalmMotion
from .kinematics import CudaKinematics
from .numerics import Numerics


CUDA=r'''
#define PALM_ID @PALM@
#define RNE_OUTPUT (NV+6)
__device__ __constant__ int hd_kind[NB];
__global__ void hd_inverse(int nw,const float* motion,const float* acceleration,const float* gravity,float* out){
  int w=blockIdx.x;if(w>=nw)return;
  float X[NB][3]={},Q[NB][4]={},F[NB][3]={},N[NB][3]={},tau[NV]={};
  for(int b=1;b<NB;b++){
    const float* m=motion+(w*NB+b)*19;float A[3];
    for(int k=0;k<3;k++){X[b][k]=m[k];A[k]=m[16+k]-gravity[k];}
    for(int k=0;k<4;k++)Q[b][k]=m[3+k];
    body_force(b,Q[b],m+7,m+13,A,c_ipos+3*b,F[b],N[b]);
  }
  for(int b=NB-1;b>=1;b--){
    if(hd_kind[b]==2){hinge_torque(b,Q[b],N[b],tau);}
    else if(hd_kind[b]==3){
      int d=c_dadr[b];float local[3];qrot_inv(Q[b],N[b],local);
      for(int k=0;k<3;k++){tau[d+k]=F[b][k];tau[d+3+k]=local[k];}
    }
    int p=c_parent[b];
    if(p>0)accumulate(X[p],X[b],F[b],N[b],F[p],N[p]);
  }
  for(int i=0;i<NV;i++)out[w*RNE_OUTPUT+i]=tau[i]+c_armature[i]*acceleration[w*NV+i];
  for(int k=0;k<3;k++){
    out[w*RNE_OUTPUT+NV+k]=F[PALM_ID][k];
    out[w*RNE_OUTPUT+NV+3+k]=N[PALM_ID][k];
  }
}
extern "C" int run(int nw,const float* motion,const float* acceleration,const float* gravity,
                    const int* meta,const float* constants,float* out){
  cudaError_t e;float *dm=nullptr,*da=nullptr,*dg=nullptr,*dout=nullptr;
  #define CHECK(call) if((e=(call))!=cudaSuccess){cudaFree(dm);cudaFree(da);cudaFree(dg);cudaFree(dout);return(int)e;}
  CHECK(cudaMemcpyToSymbol(c_parent,meta,NB*sizeof(int)));
  CHECK(cudaMemcpyToSymbol(c_dadr,meta+2*NB,NB*sizeof(int)));
  CHECK(cudaMemcpyToSymbol(hd_kind,meta+3*NB,NB*sizeof(int)));
  const float* p=constants;
  CHECK(cudaMemcpyToSymbol(c_axis,p,3*NB*sizeof(float)));p+=3*NB;
  CHECK(cudaMemcpyToSymbol(c_ipos,p,3*NB*sizeof(float)));p+=3*NB;
  CHECK(cudaMemcpyToSymbol(c_iquat,p,4*NB*sizeof(float)));p+=4*NB;
  CHECK(cudaMemcpyToSymbol(c_mass,p,NB*sizeof(float)));p+=NB;
  CHECK(cudaMemcpyToSymbol(c_inert,p,3*NB*sizeof(float)));p+=3*NB;
  CHECK(cudaMemcpyToSymbol(c_armature,p,NV*sizeof(float)));
  CHECK(cudaMalloc(&dm,(size_t)nw*NB*19*sizeof(float)));
  CHECK(cudaMalloc(&da,(size_t)nw*NV*sizeof(float)));
  CHECK(cudaMalloc(&dg,3*sizeof(float)));
  CHECK(cudaMalloc(&dout,(size_t)nw*RNE_OUTPUT*sizeof(float)));
  CHECK(cudaMemcpy(dm,motion,(size_t)nw*NB*19*sizeof(float),cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(da,acceleration,(size_t)nw*NV*sizeof(float),cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dg,gravity,3*sizeof(float),cudaMemcpyHostToDevice));
  hd_inverse<<<nw,1>>>(nw,dm,da,dg,dout);CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());
  CHECK(cudaMemcpy(out,dout,(size_t)nw*RNE_OUTPUT*sizeof(float),cudaMemcpyDeviceToHost));
  cudaFree(dm);cudaFree(da);cudaFree(dg);cudaFree(dout);return 0;
}
'''


SOLVE_CUDA=r'''
// 每环境一个完整warp，复用原引擎的稠密Cholesky与三角解。
// 这条稠密路径有逐行边界判断，区别于要求NV>=32的寄存器专用路径。
__global__ void hd_solve(int nw,const float* matrices,const float* rhs,float* outputs){
  int w=blockIdx.x,lane=threadIdx.x;
  if(w>=nw)return;
  __shared__ float M[NV*NV],b[NV];
  for(int i=lane;i<NV*NV;i+=32)M[i]=matrices[w*NV*NV+i];
  for(int i=lane;i<NV;i+=32)b[i]=rhs[w*NV+i];
  __syncwarp();chol_warp(M,lane);chol_solve_warp(M,b,lane);
  for(int i=lane;i<NV;i+=32)outputs[w*NV+i]=b[i];
}
extern "C" int run_solve(int nw,const float* matrices,const float* rhs,float* outputs){
  cudaError_t e;float *dm=nullptr,*db=nullptr,*da=nullptr;
  #define SCHECK(call) if((e=(call))!=cudaSuccess){cudaFree(dm);cudaFree(db);cudaFree(da);return(int)e;}
  SCHECK(cudaMalloc(&dm,(size_t)nw*NV*NV*sizeof(float)));
  SCHECK(cudaMalloc(&db,(size_t)nw*NV*sizeof(float)));
  SCHECK(cudaMalloc(&da,(size_t)nw*NV*sizeof(float)));
  SCHECK(cudaMemcpy(dm,matrices,(size_t)nw*NV*NV*sizeof(float),cudaMemcpyHostToDevice));
  SCHECK(cudaMemcpy(db,rhs,(size_t)nw*NV*sizeof(float),cudaMemcpyHostToDevice));
  hd_solve<<<nw,32>>>(nw,dm,db,da);SCHECK(cudaGetLastError());SCHECK(cudaDeviceSynchronize());
  SCHECK(cudaMemcpy(outputs,da,(size_t)nw*NV*sizeof(float),cudaMemcpyDeviceToHost));
  cudaFree(dm);cudaFree(db);cudaFree(da);return 0;
}
'''


@dataclass(frozen=True)
class InverseResult:
    """所需广义驱动力（不是接触力）；物体为世界力/局部力矩，hinge为关节力矩。

    palm_force与palm_moment为维持规定运动所需的世界系力/关于手掌原点的力矩，
    含手掌自身惯性与手指子树，不含自由物体的惯性（它们不连接手掌）。
    此处无接触或外力，因此也不含接触反作用。
    """
    effort: np.ndarray
    palm_force: np.ndarray
    palm_moment: np.ndarray


@dataclass(frozen=True)
class ForwardResult:
    """瞬时无接触加速度；未更新qpos/qvel。物体旋转加速度分量为物体系。

    bias包含当前规定基座加速度；residual_inf是M*qacc-(effort-bias)的最大绝对分量。
    手掌载荷由所求加速度再做逆动力学得到，无接触反作用。
    """
    qacc: np.ndarray
    effort: np.ndarray
    bias: np.ndarray
    mass_matrix: np.ndarray
    residual_inf: float
    palm_force: np.ndarray
    palm_moment: np.ndarray


class CudaDynamics:
    """无接触刚体动力学：effort=Mdd*a+b，其中b含规定基座的加速度影响。

    含重力与关节armature，不含弹簧、被动阻尼、摩擦、限位、接触、电机限幅或控制。
    默认重力取自模型；本接口不声称已支持源模型的全部物理选项。
    每实例串行使用；初始化后不跟随源模型的修改。
    """
    def __init__(self,model,layout,*,numerics="float32"):
        self.numeric=Numerics(numerics)
        self.motion=CudaKinematics(model,layout,numerics=numerics);self.layout=layout
        self.gravity=self.numeric.finite(model.opt.gravity,'gravity').copy()
        axis=np.zeros((model.nbody,3));armature=np.zeros(layout.nv)
        for b in layout.blocks:
            if b.kind=='hinge':axis[b.body_id]=model.jnt_axis[b.joint_id]
            armature[b.qvel_slice]=model.dof_armature[b.source_qvel_start:b.source_qvel_start+b.nv]
        self.constants=self.numeric.finite(np.concatenate([axis.ravel(),model.body_ipos.ravel(),model.body_iquat.ravel(),
                                                model.body_mass,model.body_inertia.ravel(),armature]),'惯性参数')
        code=source.header(model,1,1)+source.DEVICE+CUDA.replace('@PALM@',str(layout.palm_body_id))+SOLVE_CUDA
        with tempfile.TemporaryDirectory(prefix='nabla_hand_dynamics_') as tmp:
            cu,so=Path(tmp)/'dynamics.cu',Path(tmp)/'dynamics.so'
            cu.write_text(self.numeric.cuda(code));compile_cuda(cu,so,extra=('-shared','-Xcompiler','-fPIC'))
            self._library=ctypes.CDLL(str(so))
        self._run=self._library.run
        fp=self.numeric.pointer;ip=ctypes.POINTER(ctypes.c_int)
        self._run.argtypes=[ctypes.c_int,fp,fp,fp,ip,fp,fp];self._run.restype=ctypes.c_int
        self._solve=self._library.run_solve
        self._solve.argtypes=[ctypes.c_int,fp,fp,fp];self._solve.restype=ctypes.c_int

    def forward(self,state,palm,command,*,gravity=None):
        """给定手指力矩或位置PD，求无接触动态加速度；物体无外加广义力。

        每次用单位列RNE重建质量矩阵，是正确性基线，不是优化步进器。
        PD为当前时刻的显式力矩，不包含implicitfast阻尼修正或饱和。
        """
        if not isinstance(palm,PalmMotion):raise TypeError('需要显式PalmMotion')
        applied=np.zeros(self.layout.nv)
        applied[:self.layout.hand_dofs]=self.layout.desired_torque(state,command)
        M=self.numeric.finite(self.mass_matrix(state,palm),'质量矩阵')
        bias=self.inverse(state,palm,np.zeros(self.layout.nv),gravity=gravity).effort
        rhs=self.numeric.finite(applied-bias,'动力学右端')
        acceleration=np.empty(self.layout.nv,self.numeric.dtype)
        fp=self.numeric.pointer
        status=self._solve(1,M.ctypes.data_as(fp),rhs.ctypes.data_as(fp),acceleration.ctypes.data_as(fp))
        if status:raise RuntimeError(f'CUDA前向动力学失败，错误码{status}')
        if not np.isfinite(acceleration).all():raise RuntimeError('CUDA前向动力学产生非有限加速度')
        residual=float(np.max(abs(M.astype(float)@acceleration-rhs)))
        scale=float(np.max(np.sum(abs(M.astype(float)),axis=1))*np.max(abs(acceleration))+np.max(abs(rhs)))
        if residual>2e-5*scale+1e-7:
            raise RuntimeError(f'CUDA线性求解残差过大: {residual}')
        load=self.inverse(state,palm,acceleration,gravity=gravity)
        arrays=[acceleration.copy(),applied.copy(),bias.copy(),M.copy(),load.palm_force.copy(),load.palm_moment.copy()]
        for value in arrays:value.setflags(write=False)
        return ForwardResult(*arrays[:4],residual,*arrays[4:])

    def inverse(self,state,palm,qacc,*,gravity=None):
        return self.inverse_many([(state,palm,qacc)],gravity=gravity)[0]

    def inverse_many(self,cases,*,gravity=None):
        cases=list(cases)
        motions=self.motion.forward_many(cases)
        packed=self.numeric.finite([np.concatenate([m.position,m.quaternion,m.angular_velocity,m.linear_velocity,
                                          m.angular_acceleration,m.linear_acceleration],axis=1)
                          for m in motions],'刚体运动')
        acceleration=self.numeric.finite([a for _,_,a in cases],'qacc')
        g=self.numeric.finite(self.gravity if gravity is None else gravity,'gravity')
        if g.shape!=(3,):raise ValueError('gravity必须为3维世界向量')
        out=np.empty((len(cases),self.layout.nv+6),self.numeric.dtype)
        fp=self.numeric.pointer;ip=ctypes.POINTER(ctypes.c_int)
        status=self._run(len(cases),packed.ctypes.data_as(fp),acceleration.ctypes.data_as(fp),g.ctypes.data_as(fp),
                         self.motion.meta.ctypes.data_as(ip),self.constants.ctypes.data_as(fp),out.ctypes.data_as(fp))
        if status:raise RuntimeError(f'CUDA逆动力学失败，错误码{status}')
        if not np.isfinite(out).all():raise RuntimeError('CUDA逆动力学产生非有限输出')
        result=[]
        for row in out:
            parts=[row[:self.layout.nv].copy(),row[self.layout.nv:self.layout.nv+3].copy(),row[-3:].copy()]
            for part in parts:part.setflags(write=False)
            result.append(InverseResult(*parts))
        return result

    def mass_matrix(self,state,palm):
        """用单位动态加速度提取Mdd；清零所有速度、基座加速度和重力，避免大偏置相减。"""
        self.layout.validate_state(state)
        zero=HandState(state.qpos,np.zeros(self.layout.nv))
        fixed=PalmMotion.fixed(palm.position,palm.quaternion)
        columns=self.inverse_many([(zero,fixed,a) for a in np.eye(self.layout.nv)],gravity=(0,0,0))
        # 不强制对称化，以便验证能暴露真实装配误差。
        return np.column_stack([c.effort for c in columns])

    def base_acceleration_map(self,state,palm):
        """返回B，使规定基座的世界线/角加速度对动态广义力的贡献为B@[a_p,alpha_p]。

        此矩阵使用世界角加速度，不可未经转换直接等同于free joint局部旋转列。
        """
        self.layout.validate_state(state)
        zero=HandState(state.qpos,np.zeros(self.layout.nv));cases=[]
        for column in np.eye(6):
            base=PalmMotion(palm.position,palm.quaternion,[0,0,0],[0,0,0],column[:3],column[3:])
            cases.append((zero,base,np.zeros(self.layout.nv)))
        return np.column_stack([r.effort for r in self.inverse_many(cases,gravity=(0,0,0))])
