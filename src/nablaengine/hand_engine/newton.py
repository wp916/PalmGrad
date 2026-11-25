"""hand瞬时约束加速度Newton求解，保留原带保护的一维Newton线搜索。

正确性入口：CUDA目标/梯度/Hessian、原CUDA Cholesky、CUDA线搜索，Python调度。
稠密表、重新评价和串行行归约不保证与G1优化实现逐位一致；不含积分或反向。
"""
import ctypes
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import tempfile
import numpy as np
from nablaengine.cuda import compile_cuda
from nablaengine.g1_engine import source
from .dynamics import SOLVE_CUDA
from .constraint_rows import original_function
from .objective import CudaObjective
from .numerics import Numerics


@dataclass(frozen=True)
class NewtonResult:
    acceleration: np.ndarray
    evaluation: object
    iterations: int
    reason: str
    scaled_gradient: float
    warm_start_used: bool
    trace: tuple


# 算术/保护条件对应g1_engine.step.substep_post的1-D Newton块。
# 原warp行归约改成串行；2-bit packed zones改成MAXROW数组，避免544行时位移越界。
LINE_CUDA=r'''
__device__ __constant__ int line_nfloss;
#define NFLOSS line_nfloss
'''+original_function('row_piece')+'\n'+original_function('row_cost')+r'''
__global__ void line_search(int nr,const float* M,const float* vectors,const float* rows,float* result){
  const float* a=vectors;const float* afree=a+NV;const float* p=afree+NV;
  float a1=0.f,a2=0.f,rx[HAND_MAXROW],rj[HAND_MAXROW];int previous[HAND_MAXROW];
  for(int i=0;i<NV;i++){
    float ma=0.f,qsm=0.f,mp=0.f;
    for(int j=0;j<NV;j++){ma+=M[i*NV+j]*a[j];qsm+=M[i*NV+j]*afree[j];mp+=M[i*NV+j]*p[j];}
    a1+=(ma-qsm)*p[i];a2+=p[i]*mp;
  }
  for(int r=0;r<nr;r++){
    const float* row=rows+r*(NV+3);float x=0.f,jp=0.f;
    for(int j=0;j<NV;j++){x+=row[j]*a[j];jp+=row[j]*p[j];}
    rx[r]=x-row[NV+1];rj[r]=jp;previous[r]=0;
  }
  float alpha=0.f,lo=0.f,hi=1e30f,d0=0.f;bool newton=false;int reason=0,k=0;
  for(k=0;k<=40;k++){
    float fd=0.f,fdd=0.f;bool same=true;
    for(int r=0;r<nr;r++){
      const float* row=rows+r*(NV+3);float ds,dd;
      int z=row_piece(r,rx[r]+alpha*rj[r],row[NV],row[NV+2],&ds,&dd);
      same &= z==previous[r];previous[r]=z;
      fd+=ds*rj[r];fdd+=dd*rj[r]*rj[r];
    }
    if(newton && same){reason=1;break;}
    if(k==40){reason=4;break;}
    fd+=a1+alpha*a2;fdd+=a2;
    if(k==0){d0=fd;if(d0>=0.f){reason=2;break;}}
    if(fabsf(fd)<=1e-7f*fabsf(d0)){reason=3;break;}
    if(fd<0.f)lo=alpha;else hi=alpha;
    float nx=alpha-fd/fdd;newton=nx>lo && nx<hi;
    if(!newton)nx=hi<1e29f?.5f*(lo+hi):2.f*fmaxf(alpha,1e-6f);
    alpha=nx;
  }
  float derivative=a1+alpha*a2;
  for(int r=0;r<nr;r++){
    const float* row=rows+r*(NV+3);float ds,dd;
    row_piece(r,rx[r]+alpha*rj[r],row[NV],row[NV+2],&ds,&dd);derivative+=ds*rj[r];
  }
  result[0]=alpha;result[1]=k;result[2]=reason;result[3]=d0;result[4]=derivative;
}
extern "C" int run_line(int nr,int nf,const float* M,const float* vectors,const float* rows,float* output){
  float *dm=nullptr,*dv=nullptr,*dr=nullptr,*dout=nullptr;cudaError_t e;
  #define LCHECK(call) if((e=(call))!=cudaSuccess){cudaFree(dm);cudaFree(dv);cudaFree(dr);cudaFree(dout);return(int)e;}
  LCHECK(cudaMemcpyToSymbol(line_nfloss,&nf,sizeof(int)));
  LCHECK(cudaMalloc(&dm,NV*NV*sizeof(float)));LCHECK(cudaMalloc(&dv,3*NV*sizeof(float)));
  LCHECK(cudaMalloc(&dr,max(1,nr)*(NV+3)*sizeof(float)));LCHECK(cudaMalloc(&dout,5*sizeof(float)));
  LCHECK(cudaMemcpy(dm,M,NV*NV*sizeof(float),cudaMemcpyHostToDevice));
  LCHECK(cudaMemcpy(dv,vectors,3*NV*sizeof(float),cudaMemcpyHostToDevice));
  if(nr){LCHECK(cudaMemcpy(dr,rows,nr*(NV+3)*sizeof(float),cudaMemcpyHostToDevice));}
  line_search<<<1,1>>>(nr,dm,dv,dr,dout);LCHECK(cudaGetLastError());LCHECK(cudaDeviceSynchronize());
  LCHECK(cudaMemcpy(output,dout,5*sizeof(float),cudaMemcpyDeviceToHost));
  cudaFree(dm);cudaFree(dv);cudaFree(dr);cudaFree(dout);return 0;
}
'''


class CudaNewton:
    """原热启动择优、Newton方向、一维保护搜索和停止规则的hand稠密适配。

    默认停止：scale*改进<tol 或 scale*||g||2<tol，scale=1/(meaninertia*nv)。
    cost_improvement停止不等于严格动力学平衡；总是另返回scaled_gradient。
    每实例串行使用；没有tape、隐式积分、导数或批量并行接口。
    """
    def __init__(self,plan,*,numerics="float32"):
        self.numeric=Numerics(numerics)
        self.plan=plan;self.nv=plan.layout.nv;self.objective=CudaObjective(plan,numerics=numerics)
        model=SimpleNamespace(nbody=plan.layout.body_count,nq=plan.layout.nq,nv=self.nv)
        code=source.header(model,1,1)+source.DEVICE+SOLVE_CUDA+f'\n#define HAND_MAXROW {plan.max_rows}\n'+LINE_CUDA
        with tempfile.TemporaryDirectory(prefix='nabla_hand_newton_') as tmp:
            cu,so=Path(tmp)/'newton.cu',Path(tmp)/'newton.so';cu.write_text(self.numeric.cuda(code))
            compile_cuda(cu,so,extra=('-shared','-Xcompiler','-fPIC'));self._library=ctypes.CDLL(str(so))
        fp=self.numeric.pointer
        self._solve=self._library.run_solve;self._solve.argtypes=[ctypes.c_int,fp,fp,fp];self._solve.restype=ctypes.c_int
        self._line=self._library.run_line;self._line.argtypes=[ctypes.c_int,ctypes.c_int,fp,fp,fp,fp];self._line.restype=ctypes.c_int

    def linear_solve(self,matrix,right_hand_side):
        H=self.numeric.finite(matrix,'线性矩阵');rhs=self.numeric.finite(right_hand_side,'线性右端')
        if H.shape!=(self.nv,self.nv) or rhs.shape!=(self.nv,):raise ValueError('Newton线性输入形状错误')
        p=np.empty(self.nv,self.numeric.dtype);fp=self.numeric.pointer
        status=self._solve(1,*[v.ctypes.data_as(fp) for v in (H,rhs,p)])
        if status or not np.isfinite(p).all():raise RuntimeError(f'CUDA Newton Cholesky失败:{status}')
        return p

    def direction(self,hessian,gradient):
        return self.linear_solve(hessian,-np.asarray(gradient))

    def line_search(self,M,afree,a,p,table):
        # 通过objective验证模型/表结构；正式迭代内重复传输是验证实现的已知开销。
        self.objective.evaluate(M,afree,a,table)
        matrix=self.numeric.finite(M,'质量矩阵');vectors=self.numeric.finite(np.r_[a,afree,p],'线搜索向量')
        if np.asarray(p).shape!=(self.nv,):raise ValueError('方向形状错误')
        rows=self.numeric.finite(np.column_stack([table.jacobian,table.D,table.effective_aref,table.frictionloss]),'约束行')
        output=np.empty(5,self.numeric.dtype);fp=self.numeric.pointer
        status=self._line(len(table.D),table.nfrictionloss,*[v.ctypes.data_as(fp) for v in (matrix,vectors,rows,output)])
        if status or not np.isfinite(output).all():raise RuntimeError(f'CUDA线搜索失败:{status}')
        return dict(alpha=float(output[0]),iterations=int(output[1]),reason=int(output[2]),
                    initial_derivative=float(output[3]),final_derivative=float(output[4]))

    def solve(self,M,afree,table,*,warm_start=None,iterations=None,tolerance=None):
        limit=self.plan.physics['iterations'] if iterations is None else iterations
        tol=self.plan.physics['tolerance'] if tolerance is None else tolerance
        if isinstance(limit,(bool,np.bool_)) or not isinstance(limit,(int,np.integer)) or limit<0:raise ValueError('iterations必须为非负整数')
        if not np.isfinite(tol) or tol<0:raise ValueError('tolerance必须为有限非负值')
        afree=self.numeric.finite(afree,'无约束加速度');a=afree.copy()
        initial=self.objective.evaluate(M,afree,a,table);used=False
        if warm_start is not None:
            warm=self.numeric.finite(warm_start,'热启动');candidate=self.objective.evaluate(M,afree,warm,table)
            if candidate.cost<initial.cost:a=warm.copy();initial=candidate;used=True
        scale=1/(self.plan.physics['meaninertia']*self.nv);evaluation=initial
        history=[dict(iteration=0,cost=evaluation.cost,scaled_gradient=scale*float(np.linalg.norm(evaluation.gradient.astype(float))))]
        reason='iteration_limit'
        for it in range(limit):
            p=self.direction(evaluation.hessian,evaluation.gradient)
            line=self.line_search(M,afree,a,p,table)
            old=evaluation;a=self.numeric.finite(a+self.numeric.dtype(line['alpha'])*p,'Newton更新')
            evaluation=self.objective.evaluate(M,afree,a,table)
            improvement=scale*(old.cost-evaluation.cost);norm=scale*float(np.linalg.norm(evaluation.gradient.astype(float)))
            history.append(dict(iteration=it+1,cost=evaluation.cost,scaled_gradient=norm,improvement=improvement,
                                zones_changed=int(np.count_nonzero(old.zone!=evaluation.zone)),**line))
            if improvement<tol or norm<tol:
                reason='gradient' if norm<tol else 'cost_improvement';break
        a.setflags(write=False)
        return NewtonResult(a,evaluation,len(history)-1,reason,history[-1]['scaled_gradient'],used,tuple(history))
