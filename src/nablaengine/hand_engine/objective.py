"""hand加速度目标函数、梯度和分支Hessian；尚不执行Newton或时间积分。

行分段代价直接提取原G1 row_piece/row_cost，NFLOSS改为运行时表首计数。
稠密单线程CUDA实现用于正确性验证，不是最终并行/压缩工作区。
"""
import ctypes
from dataclasses import dataclass
from pathlib import Path
import tempfile
import numpy as np
from nablaengine.cuda import compile_cuda
from .constraint_rows import CUDA as SOFT_CUDA,original_function
from .constraint_table import ConstraintTable
from .numerics import Numerics


@dataclass(frozen=True)
class ObjectiveEvaluation:
    cost: float
    inertial_cost: float
    constraint_cost: float
    gradient: np.ndarray
    hessian: np.ndarray
    residual: np.ndarray
    row_force: np.ndarray
    row_curvature: np.ndarray
    row_cost: np.ndarray
    zone: np.ndarray
    generalized_force: np.ndarray


CUDA=SOFT_CUDA.split('__global__ void normal_rows')[0]+r'''
__device__ __constant__ int objective_nfloss;
#define NFLOSS objective_nfloss
'''+original_function('row_piece')+'\n'+original_function('row_cost')+r'''
__global__ void evaluate(int nr,const float* matrix,const float* afree,const float* acceleration,
                         const float* rows,float* values,float* stats){
  // values：costs[3],gradient[NV],Hessian[NV*NV],generalized_force[NV]。
  float* g=values+3;float* H=g+NV;float* force=H+NV*NV;
  float delta[NV];for(int i=0;i<NV;i++)delta[i]=acceleration[i]-afree[i];
  float inertial=0.f,total=0.f;
  for(int i=0;i<NV;i++){
    float value=0.f;for(int j=0;j<NV;j++)value+=matrix[i*NV+j]*delta[j];
    g[i]=value;inertial+=.5f*delta[i]*value;force[i]=0.f;
  }
  for(int i=0;i<NV*NV;i++)H[i]=matrix[i];
  for(int r=0;r<nr;r++){
    const float* row=rows+r*(NV+3);float x=0.f;
    for(int j=0;j<NV;j++)x+=row[j]*acceleration[j];
    x-=row[NV+1];float ds,dd;
    int z=row_piece(r,x,row[NV],row[NV+2],&ds,&dd);
    float cost=row_cost(z,x,row[NV],row[NV+2]);total+=cost;
    float* s=stats+5*r;s[0]=x;s[1]=-ds;s[2]=dd;s[3]=cost;s[4]=z;
    for(int i=0;i<NV;i++){
      g[i]+=row[i]*ds;force[i]-=row[i]*ds;
      for(int j=0;j<NV;j++)H[i*NV+j]+=dd*row[i]*row[j];
    }
  }
  values[0]=inertial+total;values[1]=inertial;values[2]=total;
}
extern "C" int run(int nr,int nf,const float* M,const float* afree,const float* acceleration,
                    const float* rows,float* values,float* stats){
  float *dm=nullptr,*df=nullptr,*da=nullptr,*dr=nullptr,*dv=nullptr,*ds=nullptr;cudaError_t e;
  #define OCHECK(call) if((e=(call))!=cudaSuccess){cudaFree(dm);cudaFree(df);cudaFree(da);cudaFree(dr);cudaFree(dv);cudaFree(ds);return(int)e;}
  OCHECK(cudaMemcpyToSymbol(objective_nfloss,&nf,sizeof(int)));
  OCHECK(cudaMalloc(&dm,NV*NV*sizeof(float)));OCHECK(cudaMalloc(&df,NV*sizeof(float)));
  OCHECK(cudaMalloc(&da,NV*sizeof(float)));OCHECK(cudaMalloc(&dr,max(1,nr)*(NV+3)*sizeof(float)));
  OCHECK(cudaMalloc(&dv,(3+2*NV+NV*NV)*sizeof(float)));OCHECK(cudaMalloc(&ds,max(1,nr)*5*sizeof(float)));
  OCHECK(cudaMemcpy(dm,M,NV*NV*sizeof(float),cudaMemcpyHostToDevice));
  OCHECK(cudaMemcpy(df,afree,NV*sizeof(float),cudaMemcpyHostToDevice));
  OCHECK(cudaMemcpy(da,acceleration,NV*sizeof(float),cudaMemcpyHostToDevice));
  if(nr) {OCHECK(cudaMemcpy(dr,rows,nr*(NV+3)*sizeof(float),cudaMemcpyHostToDevice));}
  evaluate<<<1,1>>>(nr,dm,df,da,dr,dv,ds);OCHECK(cudaGetLastError());OCHECK(cudaDeviceSynchronize());
  OCHECK(cudaMemcpy(values,dv,(3+2*NV+NV*NV)*sizeof(float),cudaMemcpyDeviceToHost));
  if(nr) {OCHECK(cudaMemcpy(stats,ds,nr*5*sizeof(float),cudaMemcpyDeviceToHost));}
  cudaFree(dm);cudaFree(df);cudaFree(da);cudaFree(dr);cudaFree(dv);cudaFree(ds);return 0;
}
'''


class CudaObjective:
    """E=.5*(a-afree)^T*M*(a-afree)+sum row_cost(Ja-aref_eff)。

    M须为已验证的对称正定动力学质量矩阵；本入口只核对形状/近似对称性。
    分支交界处沿用原row_piece的曲率选择，不承诺经典二阶导数存在。
    """
    def __init__(self,plan,*,numerics="float32"):
        self.numeric=Numerics(numerics)
        self.plan=plan;self.nv=plan.layout.nv
        with tempfile.TemporaryDirectory(prefix='nabla_hand_objective_') as tmp:
            cu,so=Path(tmp)/'objective.cu',Path(tmp)/'objective.so'
            cu.write_text(self.numeric.cuda(f'#define NV {self.nv}\n'+CUDA))
            compile_cuda(cu,so,extra=('-shared','-Xcompiler','-fPIC'));self._library=ctypes.CDLL(str(so))
        self._run=self._library.run;fp=self.numeric.pointer
        self._run.argtypes=[ctypes.c_int,ctypes.c_int]+[fp]*6;self._run.restype=ctypes.c_int

    def evaluate(self,mass_matrix,free_acceleration,acceleration,table):
        if not isinstance(table,ConstraintTable):raise TypeError('需要ConstraintTable')
        nv=self.nv;nr=len(table.D);nf=table.nfrictionloss
        self.plan.check_counts(len(table.contact_row),nr)
        M=self.numeric.finite(mass_matrix,'质量矩阵');afree=self.numeric.finite(free_acceleration,'无约束加速度');a=self.numeric.finite(acceleration,'候选加速度')
        if M.shape!=(nv,nv) or afree.shape!=(nv,) or a.shape!=(nv,):raise ValueError('动力学输入形状错误')
        if not np.allclose(M,M.T,rtol=2e-6,atol=1e-9):raise ValueError('质量矩阵必须近似对称')
        if table.jacobian.shape!=(nr,nv) or table.effective_aref.shape!=(nr,) or table.frictionloss.shape!=(nr,) or table.kind.shape!=(nr,):
            raise ValueError('约束表形状错误')
        if not isinstance(nf,(int,np.integer)) or not 0<=nf<=nr:raise ValueError('frictionloss计数错误')
        if np.any(table.kind[:nf]!=0) or np.any(table.kind[nf:]==0):raise ValueError('frictionloss行必须连续位于表首')
        if np.any(table.D<=0) or np.any(table.frictionloss[:nf]<=0) or np.any(table.frictionloss[nf:]!=0):raise ValueError('D/F约束参数错误')
        packed=self.numeric.finite(np.column_stack([table.jacobian,table.D,table.effective_aref,table.frictionloss]),'约束行')
        values=np.empty(3+2*nv+nv*nv,self.numeric.dtype);stats=np.empty((nr,5),self.numeric.dtype)
        fp=self.numeric.pointer
        status=self._run(nr,nf,*[p.ctypes.data_as(fp) for p in (M,afree,a,packed,values,stats)])
        if status:raise RuntimeError(f'CUDA目标函数失败，错误码{status}')
        if not np.isfinite(values).all() or not np.isfinite(stats).all():raise RuntimeError('目标函数产生非有限输出')
        parts=[values[3:3+nv].copy(),values[3+nv:3+nv+nv*nv].reshape(nv,nv).copy()]
        parts += [stats[:,i].copy() for i in range(4)]+[stats[:,4].astype(np.uint8),values[-nv:].copy()]
        for part in parts:part.setflags(write=False)
        return ObjectiveEvaluation(*map(float,values[:3]),*parts)
