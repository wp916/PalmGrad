"""胶囊—盒子几何方向导数实验：对现有 CUDA 分支逐运算自动微分。

设备代码由独立实验与 CudaCollisionGeometryJVP 共用。D 保存数值和一个方向导数；
候选最近点、第二接触点参数均参与微分，不使用有限差分生成梯度。
evaluate_checked 保守拒绝已知退化区；分支签名用于差分检查，仍非所有边界的完备证明。
"""
import ctypes
import re
import tempfile
from pathlib import Path
import numpy as np
from nablaengine.cuda import compile_cuda
from .collision import _collision_algorithm

DUAL=r'''
#include <cuda_runtime.h>
#include <math.h>
#include <float.h>
struct ADContext {unsigned trace;int flags;};
struct D {double v,d;ADContext*ctx;__device__ D(double x=0,double y=0,ADContext*c=0):v(x),d(y),ctx(c){}};
__device__ ADContext* context(D a,D b){return a.ctx?a.ctx:b.ctx;}
__device__ bool trace(D a,D b,bool result,int op){
 ADContext*c=context(a,b);if(c)c->trace=(c->trace ^ (unsigned)(2*op+result))*16777619u;return result;
}
__device__ D operator+(D a,D b){return D(a.v+b.v,a.d+b.d,context(a,b));}
__device__ D operator-(D a,D b){return D(a.v-b.v,a.d-b.d,context(a,b));}
__device__ D operator-(D a){return D(-a.v,-a.d,a.ctx);}
__device__ D operator*(D a,D b){return D(a.v*b.v,a.d*b.v+a.v*b.d,context(a,b));}
__device__ D operator/(D a,D b){return D(a.v/b.v,(a.d-a.v/b.v*b.d)/b.v,context(a,b));}
__device__ D& operator+=(D& a,D b){a=a+b;return a;}
__device__ D& operator-=(D& a,D b){a=a-b;return a;}
__device__ D& operator*=(D& a,D b){a=a*b;return a;}
__device__ D& operator/=(D& a,D b){a=a/b;return a;}
'''
for i,op in enumerate(('<','>','<=','>=','==','!=')):
    DUAL+=f'__device__ bool operator{op}(D a,D b){{return trace(a,b,a.v{op}b.v,{i+1});}}\n'
DUAL+=r'''
__device__ D ad_sqrt(D a){double s=sqrt(a.v);return D(s,s>0?a.d/(2*s):0,a.ctx);}
__device__ D ad_fabs(D a){return trace(a,D(0),a.v<0,7)?-a:a;}
__device__ D ad_fmin(D a,D b){return trace(a,b,a.v<b.v,8)?a:b;}
__device__ D ad_fmax(D a,D b){return trace(a,b,a.v>b.v,9)?a:b;}
__device__ D ad_fma(D a,D b,D c){return D(fma(a.v,b.v,c.v),a.d*b.v+a.v*b.d+c.d,context(a,b)?context(a,b):c.ctx);}
__device__ D ad_dot3(const D*a,const D*b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
__device__ void ad_cross(const D*a,const D*b,D*c){
 c[0]=a[1]*b[2]-a[2]*b[1];c[1]=a[2]*b[0]-a[0]*b[2];c[2]=a[0]*b[1]-a[1]*b[0];
}
struct ADS {D out[2][14];int branch;};
template<class S> __device__ void ad_add_contact(S&s,int pair,int slot,D dist,const D*p,const D*f){
 s.out[slot][0]=1;s.out[slot][1]=dist;
 for(int k=0;k<3;k++)s.out[slot][2+k]=p[k];for(int k=0;k<9;k++)s.out[slot][5+k]=f[k];
}
'''


def _function(code,name):
    m=re.search(r'__device__[^\n]*\b'+name+r'\(',code)
    if not m:raise RuntimeError(name)
    start=m.start();a=code.index('{',m.end());b=a+1;depth=1
    while depth:
        depth+=(code[b]=='{')-(code[b]=='}');b+=1
    return code[start:b]


def _patch(text,old,new,count=1):
    if text.count(old)!=count:raise RuntimeError('碰撞源码发生变化，需审核 AD 守卫注入位置：'+old)
    return text.replace(old,new)


def cuda_source():
    code=_collision_algorithm(True)
    names=('normalize3_rn','make_frame','mv3','mtv3','safe_div','sphere_box','capsule_box')
    parts=[]
    for name in names:
        part=_function(code,name)
        if name=='make_frame':
            part=_patch(part,'  double b[3]', '  if (n[0].ctx && fabs(fabs(n[1].v)-0.5)<1e-10) n[0].ctx->flags|=2;\n  double b[3]')
        if name=='sphere_box':
            part=_patch(part,'  if (dd <=', '  if (dd.ctx && dd.v<=1e-12) dd.ctx->flags|=4;\n  if (dd <=')
        if name=='capsule_box':
            part='template<class S>\n'+part
            part=_patch(part,'  bool parallel =', '  if (axis[0].ctx) { double aa=0; for(int k=0;k<3;k++)if(k!=major)aa+=axis[k].v*axis[k].v; if(aa<1e-14)axis[0].ctx->flags|=1; }\n  bool parallel =')
            part=_patch(part,'if (dist < 0.)', 'if (dist.ctx && fabs(dist.v)<1e-12) dist.ctx->flags|=8; if (dist < 0.)',2)
            part=_patch(part,'if (parallel && outside) {','if (parallel && outside) { s.branch=1;')
            part=_patch(part,'  for (int e = 0; e < 2; e++) {\n    if (e == 1',
                '  s.branch=2+(cltype+4)*2048+(clcorner+123)*8+(cledge+123);\n  for (int e = 0; e < 2; e++) {\n    if (e == 1')
        part=re.sub(r'\bdouble\b','D',part)
        for old,new in [('RFMS','ad_fms'),('RFMA','ad_fma'),('dot3_rn','ad_dot3'),('dot3','ad_dot3'),('cross_rn','ad_cross'),('sqrt','ad_sqrt'),('fabs','ad_fabs'),('fmin','ad_fmin'),('fmax','ad_fmax'),('add_contact','ad_add_contact')]+[(n,'ad_'+n) for n in names]:
            part=re.sub(r'\b'+old+r'\b',new,part)
        parts.append(part)
    return DUAL+'\n__device__ D ad_fms(D a,D b,D c){return ad_fma(-a,b,c);}\n'+'\n'.join(parts)+r'''
__global__ void kernel(const double*x,const double*dx,double*y,double*dy,int*branch){
 ADContext ctx={2166136261u,0};D a[23];for(int i=0;i<23;i++)a[i]=D(x[i],dx[i],&ctx);ADS s;s.branch=0;
 ad_capsule_box(s,0,a,a+3,a[21],a[22],a+6,a+9,a+18);
 ctx.trace=(ctx.trace ^ (unsigned)s.branch)*16777619u;
 for(int i=0;i<28;i++){y[i]=s.out[i/14][i%14].v;dy[i]=s.out[i/14][i%14].d;}*branch=(int)(ctx.trace & 0x7fffffffu);branch[1]=ctx.flags;
}
extern "C" int run(const double*x,const double*dx,double*y,double*dy,int*branch){
 double *a=0,*b=0,*c=0,*d=0;int*e=0;cudaError_t err;
 #define CHECK(call) if((err=(call))!=cudaSuccess)goto done;
 CHECK(cudaMalloc(&a,23*8));CHECK(cudaMalloc(&b,23*8));CHECK(cudaMalloc(&c,28*8));CHECK(cudaMalloc(&d,28*8));CHECK(cudaMalloc(&e,8));
 CHECK(cudaMemcpy(a,x,23*8,cudaMemcpyHostToDevice));CHECK(cudaMemcpy(b,dx,23*8,cudaMemcpyHostToDevice));
 kernel<<<1,1>>>(a,b,c,d,e);CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());
 CHECK(cudaMemcpy(y,c,28*8,cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(dy,d,28*8,cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(branch,e,8,cudaMemcpyDeviceToHost));
 done:cudaFree(a);cudaFree(b);cudaFree(c);cudaFree(d);cudaFree(e);return (int)err;
}
'''


class CapsuleBoxDirectional:
    """输入 cp3,axis_world3,bp3,R9,half3,radius,half_length，共23个标量。"""
    def __init__(self):
        with tempfile.TemporaryDirectory(prefix='capsule_box_dual_') as tmp:
            cu=Path(tmp)/'dual.cu';so=Path(tmp)/'dual.so';cu.write_text(cuda_source())
            compile_cuda(cu,so,extra=('-shared','-Xcompiler','-fPIC'))
            self.lib=ctypes.CDLL(str(so))
        self.run=self.lib.run;p=ctypes.POINTER(ctypes.c_double)
        self.run.argtypes=[p,p,p,p,ctypes.POINTER(ctypes.c_int)];self.run.restype=ctypes.c_int

    def _evaluate(self,x,direction):
        x=np.ascontiguousarray(x,dtype=np.float64);direction=np.ascontiguousarray(direction,dtype=np.float64)
        if x.shape!=(23,) or direction.shape!=(23,) or not np.isfinite(np.r_[x,direction]).all():raise ValueError('需要两个有限23维向量')
        y=np.empty((2,14));dy=np.empty_like(y);branch=(ctypes.c_int*2)();p=ctypes.POINTER(ctypes.c_double)
        status=self.run(*(a.ctypes.data_as(p) for a in (x,direction,y,dy)),branch)
        if status:raise RuntimeError(f'CUDA error {status}')
        if not np.isfinite(y).all() or not np.isfinite(dy).all():raise RuntimeError('几何自动微分出现非有限值')
        return y,dy,branch[0],branch[1]

    def evaluate(self,x,direction):
        """原始实验接口，包含分支导数；不自动接受其为经典导数。"""
        return self._evaluate(x,direction)[:3]

    def evaluate_checked(self,x,direction):
        """保守守卫：固定尺寸、合法刚体方向；已知退化区明确拒绝。"""
        x=np.asarray(x);direction=np.asarray(direction)
        if x.shape!=(23,) or direction.shape!=(23,):raise ValueError('需要两个23维向量')
        R=x[9:18].reshape(3,3);dR=direction[9:18].reshape(3,3)
        if not np.isfinite(np.r_[x,direction]).all() or not np.allclose(R.T@R,np.eye(3),atol=1e-10,rtol=0) or abs(np.linalg.det(R)-1)>1e-10 or abs(x[3:6]@x[3:6]-1)>1e-10:raise ValueError('需要有限单位轴和合法旋转矩阵')
        if np.any(x[18:]<=0) or np.any(direction[18:]!=0):raise ValueError('当前仅支持固定正尺寸')
        if abs(x[3:6]@direction[3:6])>1e-10 or not np.allclose(R.T@dR+dR.T@R,0,atol=1e-10,rtol=0):raise ValueError('方向不是合法刚体一阶扰动')
        y,dy,branch,flags=self._evaluate(x,direction)
        if flags:
            names=[name for bit,name in ((1,'轴平行/近轴平行'),(2,'接触架切换'),(4,'内部/近零球盒距离'),(8,'接触激活边界')) if flags&bit]
            raise ValueError('胶囊盒子几何导数保守拒绝：'+','.join(names))
        return y,dy,branch
