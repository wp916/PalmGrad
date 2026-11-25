"""Hand基本形状GPU碰撞：复用G1分支算法，双精度几何，CPU仅打包/解码。

显式pair限定，支持plane-sphere/plane-capsule/sphere-sphere/sphere-capsule/
capsule-capsule。enable_boxes=True额外开启plane/sphere/capsule对box的前向，
不支持box-box、cylinder或mesh；盒子几何反向支持sphere-box及保守守卫允许的capsule-box。
不会自动枚举隐式碰撞对。
每实例复用独立显存工作区，调用串行化；构造后模型视为冻结。
"""
import ctypes
from pathlib import Path
import re
import tempfile
import threading
import numpy as np
from nablaengine.cuda import compile_cuda
from nablaengine.g1_engine.step import STEP_DEVICE
from nablaengine.model.hand import PalmMotion
from .contact_precision import CUDA as POSES
from .contacts import ContactFrame
from .layout import build_topology


def _original_function(name):
    # 精确匹配定义，避免注释中make_frame(...)被误认成函数入口。
    match=re.search(r'__device__ __forceinline__ [^\n]+?\b'+name+r'\(',STEP_DEVICE)
    if match is None:raise RuntimeError(f'找不到原碰撞函数{name}')
    brace=STEP_DEVICE.index('{',match.end());end=brace+1;depth=1
    while depth:
        depth+=(STEP_DEVICE[end]=='{')-(STEP_DEVICE[end]=='}');end+=1
    return STEP_DEVICE[match.start():end]


def _collision_algorithm(include_boxes=False):
    names=('normalize3_rn','make_frame','sphere_sphere','plane_point',
           'plane_capsule_frame','capsule_end','segment_point')
    code='\n'.join(_original_function(n) for n in names)
    start=STEP_DEVICE.index('struct CapCap {')
    end=STEP_DEVICE.index('// 3x3 helpers',start)
    code+='\n'+STEP_DEVICE[start:end]
    if include_boxes:
        # 复用原sphere_box/capsule_box；box-box另行验证，不在本阶段开放。
        start=STEP_DEVICE.index('// 3x3 helpers')
        end=STEP_DEVICE.index('__device__ __forceinline__ void rotmore_of',start)
        code+='\n'+STEP_DEVICE[start:end]
        # det的量纲为长度四次方；固定1e-15会把真实小倾角当作平行。
        # 只改变退化判断，保留求最近点、接触力和积分公式。
        old='if (fabsf(det) < 1e-15f) continue;'
        if code.count(old)!=1:raise RuntimeError('capsule_box行列式源码发生变化，请审核适配')
        code=code.replace(old,'if (fabsf(det) < 64 * DBL_EPSILON * ma * mc) continue;')
        # 轴平行退化情形：面/边上的最近点是整段而非单点。
        # 直接求该段两端，避免原角点方向位受理论零分量的舍入符号影响。
        # 非平行情形保留原候选算法；不改实际axis/ha、球盒距离或接触力公式。
        marker='  int axisdir = int(ha[0] > 0.f)'
        at=code.index(marker)
        code=code[:at]+r'''
  int major = 0;
  for (int k = 1; k < 3; k++) if (fabsf(axis[k]) > fabsf(axis[major])) major = k;
  bool parallel = true, outside = false;
  for (int k = 0; k < 3; k++) if (k != major) {
    parallel = parallel && fabsf(axis[k]) <= 8 * DBL_EPSILON;
    outside = outside || fabsf(pos[k]) > bs[k];
  }
  if (parallel && outside) {
    float lo = fmaxf(-hl, fminf((-bs[major]-pos[major])/axis[major], (bs[major]-pos[major])/axis[major]));
    float hi = fminf(hl, fmaxf((-bs[major]-pos[major])/axis[major], (bs[major]-pos[major])/axis[major]));
    if (lo <= hi) {
      for (int e = 0; e < 2; e++) {
        if (e == 1 && hi-lo <= 32 * DBL_EPSILON * hl) break;
        float along = e ? hi : lo, lc[3], world[3], ce[3], dist, pc[3], n[3], frame[9];
        for (int k = 0; k < 3; k++) lc[k] = pos[k] + axis[k] * along;
        mv3(R, lc, world);
        for (int k = 0; k < 3; k++) ce[k] = bp[k] + world[k];
        sphere_box(ce, cr, bp, R, bs, &dist, pc, n);
        if (dist < 0.f) { make_frame(n, frame); add_contact(s, pair, e, dist, pc, frame); }
      }
      return;
    }
  }
'''+code[at:]
    start=STEP_DEVICE.index('template <class S>\n__device__ DK_INL_COLLIDE void collide_pair')
    end=STEP_DEVICE.index('// x^p for the solimp',start)
    code+='\n'+STEP_DEVICE[start:end]
    if include_boxes:
        start=code.index('  if (t1 == 3 && t2 == 3)')
        end=code.index('\n#endif',start)
        code=code[:start]+code[end:]
        at=code.index('  if (t1 == 2 && t2 == 3)')
        code=code[:at]+'''  if (t1 == 1 && t2 == 3) {
    float R[9], dist, pos[3], n[3], frame[9];
    quat2mat(q2, R); sphere_box(p1, sz1[0], p2, R, sz2, &dist, pos, n);
    if (dist < 0.f) { make_frame(n, frame); add_contact(s, pair, 0, dist, pos, frame); }
    return;
  }
'''+code[at:]
    # 原unsupported是共享计数器；此适配器单线程且构造时已审核组合。
    code=code.replace('atomicOr(&s.unsupported, 1)', 's.unsupported = 1')
    # 除上述显式box数值修补外，转换精度时保留原阈值、正则项和候选策略。
    code=re.sub(r'\bfloat\b','double',code)
    code=re.sub(r'\b(sqrt|fabs|fmin|fmax)f\b',r'\1',code)
    code=re.sub(r'(\d(?:[\d.eE+-]*))f\b',r'\1',code)
    return code


HEADER=r'''
#include <cuda_runtime.h>
#include <math.h>
#include <float.h>
#ifndef HAS_BOX
#define HAS_BOX 0
#endif
#define SLOTS 8
#define DK_INL_COLLIDE __forceinline__
#define RADD(a,b) __dadd_rn((a),(b))
#define RSUB(a,b) __dsub_rn((a),(b))
#define RMUL(a,b) __dmul_rn((a),(b))
#define RFMA(a,b,c) __fma_rn((a),(b),(c))
#define RFMS(a,b,c) __fma_rn(-(a),(b),(c))
__device__ __constant__ int c_parent[NB],c_qadr[NB],jc_kind[NB];
__device__ __constant__ int c_gtype[NG],c_gbody[NG],pairs[2*NP];
__device__ __constant__ double c_gsize[3*NG],c_grbound[NG],geom[7*NG];
#define PG1(p) pairs[2*(p)]
#define PG2(p) pairs[2*(p)+1]
'''
HELPERS=r'''
#define dot3_rn cp_dot
#define dot3 cp_dot
#define qrot cp_qrot
#define cross_rn cp_cross
__device__ void zaxis_rn(const double* q,double* out){double z[3]={0.,0.,1.};cp_qrot(q,z,out);}
struct CollisionState {double gx[NG][3],gq[NG][4];double* out;int unsupported;};
template<class S> __device__ bool add_contact(S& s,int pair,int slot,double dist,const double* pos,const double* frame){
  double* o=s.out+(pair*SLOTS+slot)*14;o[0]=1.;o[1]=dist;
  for(int k=0;k<3;k++)o[2+k]=pos[k];for(int k=0;k<9;k++)o[5+k]=frame[k];return true;
}
template<class S> __device__ void add_sphere_contact(S& s,int pair,int slot,double nu,double dist,const double* pos,const double* n);
'''
SPHERE=r'''
template<class S> __device__ void add_sphere_contact(S& s,int pair,int slot,double nu,double dist,const double* pos,const double* n){
 double frame[9];make_frame(n,frame);add_contact(s,pair,slot,dist,pos,frame);
}
'''
TAIL=r'''
__global__ void collide(const double* poses,double* out){
 CollisionState s;s.out=out;s.unsupported=0;
 for(int g=0;g<NG;g++){
   const double* p=poses+7*c_gbody[g];double offset[3];cp_qrot(p+3,geom+7*g,offset);
   for(int k=0;k<3;k++)s.gx[g][k]=p[k]+offset[k];cp_qmul(p+3,geom+7*g+3,s.gq[g]);
 }
 for(int pair=0;pair<NP;pair++)collide_pair(s,pair);
}
extern "C" int run(const int* meta,const int* gm,const double* fixed,const double* input,double* output){
 cudaError_t e;double *di=nullptr,*dp=nullptr,*dout=nullptr;
 #define CHECK(call) if((e=(call))!=cudaSuccess){cudaFree(di);cudaFree(dp);cudaFree(dout);return(int)e;}
 CHECK(cudaMemcpyToSymbol(c_parent,meta,NB*sizeof(int)));
 CHECK(cudaMemcpyToSymbol(c_qadr,meta+NB,NB*sizeof(int)));
 CHECK(cudaMemcpyToSymbol(jc_kind,meta+3*NB,NB*sizeof(int)));
 CHECK(cudaMemcpyToSymbol(c_gtype,gm,NG*sizeof(int)));
 CHECK(cudaMemcpyToSymbol(c_gbody,gm+NG,NG*sizeof(int)));
 CHECK(cudaMemcpyToSymbol(pairs,gm+2*NG,2*NP*sizeof(int)));
 CHECK(cudaMemcpyToSymbol(c_gsize,fixed,3*NG*sizeof(double)));
 CHECK(cudaMemcpyToSymbol(c_grbound,fixed+3*NG,NG*sizeof(double)));
 CHECK(cudaMemcpyToSymbol(geom,fixed+4*NG,7*NG*sizeof(double)));
 CHECK(cudaMalloc(&di,(NQ+7+10*NB)*sizeof(double)));
 CHECK(cudaMalloc(&dp,7*NB*sizeof(double)));CHECK(cudaMalloc(&dout,NP*4*14*sizeof(double)));
 CHECK(cudaMemcpy(di,input,(NQ+7+10*NB)*sizeof(double),cudaMemcpyHostToDevice));
 CHECK(cudaMemset(dout,0,NP*4*14*sizeof(double)));
 cp_poses<<<1,1>>>(di,dp);CHECK(cudaGetLastError());collide<<<1,1>>>(dp,dout);
 CHECK(cudaGetLastError());CHECK(cudaDeviceSynchronize());
 CHECK(cudaMemcpy(output,dout,NP*4*14*sizeof(double),cudaMemcpyDeviceToHost));
 cudaFree(di);cudaFree(dp);cudaFree(dout);return 0;
}
'''

# 保留上面的临时分配run作为回归参考；正常调用仅使用以下常驻工作区。
WORKSPACE=r'''
struct Workspace {double *input=nullptr,*poses=nullptr,*output=nullptr;};
extern "C" int destroy_workspace(void* handle){
 if(!handle)return 0;Workspace* w=(Workspace*)handle;cudaError_t first=cudaSuccess,e;
 e=cudaFree(w->input);if(first==cudaSuccess)first=e;
 e=cudaFree(w->poses);if(first==cudaSuccess)first=e;
 e=cudaFree(w->output);if(first==cudaSuccess)first=e;
 delete w;return (int)first;
}
extern "C" int create_workspace(const int* meta,const int* gm,const double* fixed,const double* input,void** handle){
 *handle=nullptr;Workspace* w=new Workspace;cudaError_t e;
 #define WCHECK(call) if((e=(call))!=cudaSuccess){destroy_workspace(w);return(int)e;}
 WCHECK(cudaMemcpyToSymbol(c_parent,meta,NB*sizeof(int)));
 WCHECK(cudaMemcpyToSymbol(c_qadr,meta+NB,NB*sizeof(int)));
 WCHECK(cudaMemcpyToSymbol(jc_kind,meta+3*NB,NB*sizeof(int)));
 WCHECK(cudaMemcpyToSymbol(c_gtype,gm,NG*sizeof(int)));
 WCHECK(cudaMemcpyToSymbol(c_gbody,gm+NG,NG*sizeof(int)));
 WCHECK(cudaMemcpyToSymbol(pairs,gm+2*NG,2*NP*sizeof(int)));
 WCHECK(cudaMemcpyToSymbol(c_gsize,fixed,3*NG*sizeof(double)));
 WCHECK(cudaMemcpyToSymbol(c_grbound,fixed+3*NG,NG*sizeof(double)));
 WCHECK(cudaMemcpyToSymbol(geom,fixed+4*NG,7*NG*sizeof(double)));
 WCHECK(cudaMalloc(&w->input,(NQ+7+10*NB)*sizeof(double)));
 WCHECK(cudaMalloc(&w->poses,7*NB*sizeof(double)));
 WCHECK(cudaMalloc(&w->output,NP*4*14*sizeof(double)));
 // 初始化时上传完整输入；以后只覆盖qpos和palm，保留尾部静态刚体数据。
 WCHECK(cudaMemcpy(w->input,input,(NQ+7+10*NB)*sizeof(double),cudaMemcpyHostToDevice));
 *handle=w;return 0;
}
extern "C" int run_workspace(void* handle,const double* input,double* output){
 if(!handle)return (int)cudaErrorInvalidValue;Workspace* w=(Workspace*)handle;cudaError_t e;
 #define RCHECK(call) if((e=(call))!=cudaSuccess)return(int)e;
 RCHECK(cudaMemcpy(w->input,input,(NQ+7)*sizeof(double),cudaMemcpyHostToDevice));
 RCHECK(cudaMemset(w->output,0,NP*4*14*sizeof(double)));
 cp_poses<<<1,1>>>(w->input,w->poses);RCHECK(cudaGetLastError());
 collide<<<1,1>>>(w->poses,w->output);RCHECK(cudaGetLastError());RCHECK(cudaDeviceSynchronize());
 RCHECK(cudaMemcpy(output,w->output,NP*4*14*sizeof(double),cudaMemcpyDeviceToHost));
 return 0;
}
'''


class CudaPrimitiveContacts:
    """可直接作为CudaImplicitFastStepper的contact_provider，无MuJoCo运行时调用。

    max_contacts为接触点容量，超限报错而非截断。只计算显式pair；要求所有
    geom的contype/conaffinity均为0，避免误漏用户期望的隐式碰撞对。
    enable_boxes显式开启盒子前向；每pair预留8个候选槽以覆盖plane-box。
    """
    def __init__(self,model,layout,*,max_contacts=128,enable_boxes=False):
        self._lock=threading.Lock();self._closed=False;self._handle=ctypes.c_void_p()
        if not isinstance(enable_boxes,(bool,np.bool_)):raise TypeError('enable_boxes必须为bool')
        if isinstance(max_contacts,bool) or not isinstance(max_contacts,(int,np.integer)) or max_contacts<1:
            raise ValueError('max_contacts必须是正整数')
        self.layout=layout;self.max_contacts=int(max_contacts)
        topology=build_topology(model,layout);self.meta=topology.meta
        self.nb=int(model.nbody);self.ng=int(model.ngeom);self.npair=int(model.npair)
        if np.any(model.geom_contype) or np.any(model.geom_conaffinity):
            raise ValueError('GPU基本形状碰撞仅支持显式pair，请关闭geom隐式碰撞掩码')
        if np.any(model.pair_margin) or np.any(model.pair_gap):raise ValueError('仅支持零pair margin/gap')
        # MuJoCo原生类型：plane=0,sphere=2,capsule=3；G1内部：0,1,2。
        types=np.array([{0:0,2:1,3:2}.get(int(t),-1) for t in model.geom_type],np.int32)
        if enable_boxes:types[np.asarray(model.geom_type)==6]=3
        pair=np.c_[model.pair_geom1,model.pair_geom2].astype(np.int32)
        self.reverse=np.zeros(self.npair,bool)
        for p,(a,b) in enumerate(pair):
            if types[a]<0 or types[b]<0 or (types[a]==0 and types[b]==0):
                raise ValueError(f'pair {p}的形状组合尚未接入GPU基本形状碰撞')
            if types[a]==3 and types[b]==3:raise ValueError('box-box尚未接入验证，禁止静默跳过')
            if any(types[g]==2 and model.geom_size[g,1]<=0 for g in (a,b)):
                raise ValueError(f'pair {p}胶囊半长必须大于0')
            if types[a]>types[b]:pair[p]=[b,a];self.reverse[p]=True
        self.bodies=np.c_[model.geom_bodyid[model.pair_geom1],model.geom_bodyid[model.pair_geom2]].copy()
        self.gmeta=np.ascontiguousarray(np.r_[types,model.geom_bodyid,pair.ravel()],dtype=np.int32)
        self.fixed=np.ascontiguousarray(np.r_[model.geom_size.ravel(),model.geom_rbound,
                    np.c_[model.geom_pos,model.geom_quat].ravel()],dtype=np.float64)
        self.geometry=np.ascontiguousarray(np.r_[model.body_pos.ravel(),model.body_quat.ravel(),topology.axes.ravel()],dtype=np.float64)
        for a in (self.bodies,self.reverse,self.gmeta,self.fixed,self.geometry):a.setflags(write=False)
        self.calls=0;self.counts=[]
        if not self.npair:return
        algorithm=_collision_algorithm(include_boxes=enable_boxes)
        # add_sphere_contact需要make_frame的定义，因此插到collide_pair之前。
        at=algorithm.index('template <class S>')
        algorithm=algorithm[:at]+SPHERE+algorithm[at:]
        code=f'#define NB {self.nb}\n#define NQ {layout.nq}\n#define NG {self.ng}\n#define NP {self.npair}\n#define HAS_BOX {int(bool(enable_boxes))}\n'+HEADER+POSES+HELPERS+algorithm+TAIL.replace('NP*4*14','NP*SLOTS*14')+WORKSPACE.replace('NP*4*14','NP*SLOTS*14')
        with tempfile.TemporaryDirectory(prefix='nabla_hand_collision_') as tmp:
            cu,so=Path(tmp)/'collision.cu',Path(tmp)/'collision.so'
            cu.write_text(code);compile_cuda(cu,so,extra=('-shared','-Xcompiler','-fPIC'))
            self._library=ctypes.CDLL(str(so))
        self._run=self._library.run
        ip=ctypes.POINTER(ctypes.c_int);dp=ctypes.POINTER(ctypes.c_double)
        self._run.argtypes=[ip,ip,dp,dp,dp];self._run.restype=ctypes.c_int
        self._create=self._library.create_workspace
        self._create.argtypes=[ip,ip,dp,dp,ctypes.POINTER(ctypes.c_void_p)];self._create.restype=ctypes.c_int
        self._execute=self._library.run_workspace
        self._execute.argtypes=[ctypes.c_void_p,dp,dp];self._execute.restype=ctypes.c_int
        self._destroy=self._library.destroy_workspace
        self._destroy.argtypes=[ctypes.c_void_p];self._destroy.restype=ctypes.c_int
        initial=np.zeros(layout.nq+7+10*self.nb,np.float64);initial[layout.nq+7:]=self.geometry
        status=self._create(self.meta.ctypes.data_as(ip),self.gmeta.ctypes.data_as(ip),self.fixed.ctypes.data_as(dp),
                            initial.ctypes.data_as(dp),ctypes.byref(self._handle))
        if status:raise RuntimeError(f'CUDA碰撞工作区创建失败，错误码{status}')

    @property
    def closed(self):
        return self._closed

    @property
    def workspace_bytes(self):
        """本实例申请的GPU输入、位姿、接触输出缓冲区字节数；不含CUDA模块等开销。"""
        return 0 if self._closed or not self.npair else 8*(self.layout.nq+7+17*self.nb+56*self.npair)

    def close(self):
        """幂等释放工作区；释放后不能继续调用。推荐使用with确定释放时机。"""
        with self._lock:
            if self._closed:return
            self._closed=True
            if self._handle.value:
                handle=self._handle;self._handle=ctypes.c_void_p()
                status=self._destroy(handle)
                if status:raise RuntimeError(f'CUDA碰撞工作区释放失败，错误码{status}')

    def __enter__(self):
        if self._closed:raise RuntimeError('CUDA碰撞工作区已关闭')
        return self

    def __exit__(self,*args):
        self.close()

    def __del__(self):
        # 构造失败或解释器退出时尽力释放；显式close仍会报告CUDA释放错误。
        try:self.close()
        except Exception:pass

    def __call__(self,state,palm):
        with self._lock:
            if self._closed:raise RuntimeError('CUDA碰撞工作区已关闭')
            return self._evaluate(state,palm)

    def evaluate_with_slots(self,state,palm):
        """返回contacts/pair_ids/slots；供反向精确对应原候选点，不改变普通调用。"""
        with self._lock:
            if self._closed:raise RuntimeError('CUDA碰撞工作区已关闭')
            return self._evaluate(state,palm,include_slots=True)

    def _evaluate(self,state,palm,*,include_slots=False):
        self.layout.validate_state(state)
        if not isinstance(palm,PalmMotion):raise TypeError('需要PalmMotion')
        if not self.npair:
            self.calls+=1;self.counts.append(0);return ([],[],[]) if include_slots else ([],[])
        packed=np.ascontiguousarray(np.r_[state.qpos,palm.position,palm.quaternion],dtype=np.float64)
        out=np.empty((self.npair,8,14),np.float64)
        dp=ctypes.POINTER(ctypes.c_double)
        status=self._execute(self._handle,packed.ctypes.data_as(dp),out.ctypes.data_as(dp))
        if status:raise RuntimeError(f'CUDA碰撞失败，错误码{status}')
        if not np.isfinite(out).all():raise RuntimeError('CUDA碰撞产生非有限输出')
        count=int(np.count_nonzero(out[:,:,0]))
        if count>self.max_contacts:raise ValueError(f'实际接触点{count}超过容量{self.max_contacts}，未截断')
        contacts=[];pair_ids=[];slots=[]
        for p in range(self.npair):
            for slot,row in enumerate(out[p]):
                if row[0]==0:continue
                frame=row[5:].reshape(3,3).copy()
                if self.reverse[p]:frame[[0,2]]*=-1
                try:c=ContactFrame(int(self.bodies[p,0]),int(self.bodies[p,1]),row[2:5],frame,row[1])
                except ValueError as exc:
                    raise ValueError(f'pair {p}原几何输出不满足接触架约束；未自动改写原算法: {exc}') from exc
                contacts.append(c);pair_ids.append(p);slots.append(slot)
        self.calls+=1;self.counts.append(count)
        return (contacts,pair_ids,slots) if include_slots else (contacts,pair_ids)
