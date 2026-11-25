"""基本形状碰撞的解析几何JVP：复用原G1几何VJP，以13个输出基向量收集。

原基本体复用G1 VJP，球—盒子使用解析VJP，胶囊—盒子使用逐运算AD。
几何前向重放保持一致；胶囊盒子仅开放保守守卫允许的分支。
原基本体normal/parallel采用精确分支导数；胶囊盒子退化区保守拒绝。
只支持动态状态位置扰动，规定手掌固定；plane必须附着world或规定手掌。
逐方向正确性基线，不是最终融合批量性能实现。
"""
import ctypes,re,tempfile
from pathlib import Path
from types import SimpleNamespace
from dataclasses import dataclass
import numpy as np
from nablaengine.cuda import compile_cuda
from nablaengine.g1_engine import source
from nablaengine.g1_engine.adjoint_contact import DEVICE
from .collision import _collision_algorithm,CudaPrimitiveContacts
from .contact_position import CUDA as CONTACT_CUDA
from .numerics import Numerics
from .box_geometry_adjoint import CUDA as BOX_VJP
from .capsule_box_directional import cuda_source as capsule_box_source


def _original(name):
    match=re.search(r'__device__ (?:__forceinline__|__noinline__) [^\n]+?\b'+name+r'\(',DEVICE)
    if match is None:raise RuntimeError(f'原几何反向函数缺失:{name}')
    brace=DEVICE.index('{',match.end());end=brace+1;depth=1
    while depth:
        depth+=(DEVICE[end]=='{')-(DEVICE[end]=='}');end+=1
    return DEVICE[match.start():end]


PREAMBLE=r'''
#define DK_ADJ_NORMAL 0
#define DK_ADJ_PARALLEL 0
#define ADJ_BR_FREE1 1
#define ADJ_BR_FREE2 2
#define ADJ_BR_PARALLEL 4
#define ADJ_BR_FALLBACK 8
#define ADJ_BR_FRAME_Z 16
'''
KERNEL=r'''
__global__ void geometry_jvp(int nc,const int* contacts,const int* gm,const double* fixed,const double* poses,
                            const double* delta,double* replay,double* output,int* branches){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=13*nc)return;int c=i/13,k=i%13;
 int pair=contacts[3*c],slot=contacts[3*c+1],reverse=contacts[3*c+2];
 int g1=gm[2*NG+2*pair],g2=gm[2*NG+2*pair+1],b1=gm[NG+g1],b2=gm[NG+g2];
 const double* stat=fixed+4*NG;double gx[2][3],gq[2][4],dx[2][3];int bs[2]={b1,b2},gs[2]={g1,g2};
 for(int j=0;j<2;j++){
  const double* p=poses+7*bs[j];const double* d=delta+6*bs[j];double offset[3],rot[3];
  cp_qrot(p+3,stat+7*gs[j],offset);cp_qmul(p+3,stat+7*gs[j]+3,gq[j]);cp_cross(d+3,offset,rot);
  for(int a=0;a<3;a++){gx[j][a]=p[a]+offset[a];dx[j][a]=d[a]+rot[a];}
 }
 double dist,pos[3],fr[9],nu,cot[13]={},gb[12];
 bool flip=reverse&&((k>=4&&k<7)||k>=10);cot[k]=flip?-1:1;
 int br=hand_contact_geom(gm[g1],gm[g2],slot,gx[0],gq[0],fixed+3*g1,gx[1],gq[1],fixed+3*g2,&dist,pos,fr,&nu,cot,gb);
 double sum=0;for(int j=0;j<2;j++)for(int a=0;a<3;a++)sum+=gb[6*j+a]*dx[j][a]+gb[6*j+3+a]*delta[6*bs[j]+3+a];
 output[i]=sum;double value=k==0?dist:(k<4?pos[k-1]:fr[k-4]);replay[i]=flip?-value:value;
 if(k==0){branches[2*c]=br;branches[2*c+1]=(gm[g1]>0&&nu==0)?1:0;}
}
extern "C" int run(int nc,const int* active,const int* gm,const double* fixed,const double* exact,const int* meta,
                    const double* direction,double* replay,double* output,int* branches){
 double *df=nullptr,*de=nullptr,*dd=nullptr,*dp=nullptr,*dt=nullptr,*dr=nullptr,*do_=nullptr;int *dc=nullptr,*dg=nullptr,*db=nullptr;cudaError_t e;
 #define GC(call) if((e=(call))!=cudaSuccess){cudaFree(df);cudaFree(de);cudaFree(dd);cudaFree(dp);cudaFree(dt);cudaFree(dr);cudaFree(do_);cudaFree(dc);cudaFree(dg);cudaFree(db);return(int)e;}
 GC(cudaMemcpyToSymbol(c_parent,meta,NB*sizeof(int)));GC(cudaMemcpyToSymbol(c_qadr,meta+NB,NB*sizeof(int)));
 GC(cudaMemcpyToSymbol(c_dadr,meta+2*NB,NB*sizeof(int)));GC(cudaMemcpyToSymbol(jc_kind,meta+3*NB,NB*sizeof(int)));
 GC(cudaMalloc(&df,11*NG*sizeof(double)));GC(cudaMalloc(&de,(NQ+7+10*NB)*sizeof(double)));GC(cudaMalloc(&dd,NV*sizeof(double)));
 GC(cudaMalloc(&dp,7*NB*sizeof(double)));GC(cudaMalloc(&dt,6*NB*sizeof(double)));GC(cudaMalloc(&dr,nc*13*sizeof(double)));GC(cudaMalloc(&do_,nc*13*sizeof(double)));
 GC(cudaMalloc(&dc,nc*3*sizeof(int)));GC(cudaMalloc(&dg,(2*NG+2*NP)*sizeof(int)));GC(cudaMalloc(&db,2*nc*sizeof(int)));
 GC(cudaMemcpy(df,fixed,11*NG*sizeof(double),cudaMemcpyHostToDevice));GC(cudaMemcpy(de,exact,(NQ+7+10*NB)*sizeof(double),cudaMemcpyHostToDevice));GC(cudaMemcpy(dd,direction,NV*sizeof(double),cudaMemcpyHostToDevice));
 GC(cudaMemcpy(dc,active,3*nc*sizeof(int),cudaMemcpyHostToDevice));GC(cudaMemcpy(dg,gm,(2*NG+2*NP)*sizeof(int),cudaMemcpyHostToDevice));
 cp_poses<<<1,1>>>(de,dp);pose_tangents<<<1,1>>>(de,dp,dd,dt);geometry_jvp<<<(nc*13+63)/64,64>>>(nc,dc,dg,df,dp,dt,dr,do_,db);
 GC(cudaGetLastError());GC(cudaDeviceSynchronize());GC(cudaMemcpy(replay,dr,13*nc*sizeof(double),cudaMemcpyDeviceToHost));GC(cudaMemcpy(output,do_,13*nc*sizeof(double),cudaMemcpyDeviceToHost));GC(cudaMemcpy(branches,db,2*nc*sizeof(int),cudaMemcpyDeviceToHost));
 cudaFree(df);cudaFree(de);cudaFree(dd);cudaFree(dp);cudaFree(dt);cudaFree(dr);cudaFree(do_);cudaFree(dc);cudaFree(dg);cudaFree(db);return 0;
}
'''


CAPSULE_HELPER=r'''
// 原几何 AD 的设备实现直接接入相同位姿/切空间约定，无 MuJoCo 或数值差分。
__device__ int capsule_geometry(int slot,int k,const double* p1,const double* q1,const double* size1,
 const double* p2,const double* q2,const double* size2,const double* dp1,const double* w1,
 const double* dp2,const double* w2,double* value,double* slope,int* flags,int* any_active){
 double R[9],axis[3],z[3]={0,0,1},da[3],dR[9];quat2mat(q2,R);cp_qrot(q1,z,axis);cp_cross(w1,axis,da);
 for(int col=0;col<3;col++){
  double a[3]={R[col],R[3+col],R[6+col]},b[3];cp_cross(w2,a,b);
  for(int row=0;row<3;row++)dR[3*row+col]=b[row];
 }
 ADContext ctx={2166136261u,0};D a[23];
 for(int j=0;j<3;j++){
  a[j]=D(p1[j],dp1[j],&ctx);a[3+j]=D(axis[j],da[j],&ctx);a[6+j]=D(p2[j],dp2[j],&ctx);a[18+j]=D(size2[j],0,&ctx);
 }
 for(int j=0;j<9;j++)a[9+j]=D(R[j],dR[j],&ctx);
 a[21]=D(size1[0],0,&ctx);a[22]=D(size1[1],0,&ctx);ADS state;state.branch=0;
 ad_capsule_box(state,0,a,a+3,a[21],a[22],a+6,a+9,a+18);
 ctx.trace=(ctx.trace ^ (unsigned)state.branch)*16777619u;
 *flags=ctx.flags;*any_active=state.out[0][0].v!=0 || state.out[1][0].v!=0;
 if(slot<0||slot>=2){*value=0;*slope=0;return -3;}
 *value=state.out[slot][k+1].v;*slope=state.out[slot][k+1].d;
 return (int)(ctx.trace & 0x7fffffffu);
}
__global__ void capsule_guards(const int* gm,const double* fixed,const double* poses,int* flags){
 int pair=blockIdx.x*blockDim.x+threadIdx.x;if(pair>=NP)return;flags[pair]=0;
 int g1=gm[2*NG+2*pair],g2=gm[2*NG+2*pair+1];if(gm[g1]!=2||gm[g2]!=3)return;
 const double* stat=fixed+4*NG;double gx[2][3],gq[2][4],zero[3]={};int gs[2]={g1,g2};
 for(int j=0;j<2;j++){
  const double* p=poses+7*gm[NG+gs[j]];double off[3];cp_qrot(p+3,stat+7*gs[j],off);cp_qmul(p+3,stat+7*gs[j]+3,gq[j]);
  for(int k=0;k<3;k++)gx[j][k]=p[k]+off[k];
 }
 double v,d;int f,active;capsule_geometry(0,0,gx[0],gq[0],fixed+3*g1,gx[1],gq[1],fixed+3*g2,zero,zero,zero,zero,&v,&d,&f,&active);
 // 无接触分支可正常返回零导数；距离刚好为0仍需检测，不能因无激活行而漏检。
 flags[pair]=active?f:(f&8);
}
'''
CAPSULE_DISPATCH=r'''
 if(gm[g1]==2&&gm[g2]==3){
  double value,slope;int flags,active;
  int br=capsule_geometry(slot,k,gx[0],gq[0],fixed+3*g1,gx[1],gq[1],fixed+3*g2,dx[0],delta+6*b1+3,dx[1],delta+6*b2+3,&value,&slope,&flags,&active);
  bool flip=reverse&&((k>=4&&k<7)||k>=10);replay[i]=flip?-value:value;output[i]=flip?-slope:slope;
  if(k==0){branches[2*c]=br;branches[2*c+1]=0;}return;
 }
'''


def _patch_kernel(code,old,new,count=1):
    if code.count(old)!=count:raise RuntimeError('几何 JVP 内核发生变化，需审核胶囊分派适配：'+old)
    return code.replace(old,new)


@dataclass(frozen=True)
class CollisionGeometryJVP:
    contacts: tuple
    pair_ids: np.ndarray
    slots: np.ndarray
    branch: np.ndarray
    distance: np.ndarray
    position: np.ndarray
    frame: np.ndarray
    replay_error: float


class CudaCollisionGeometryJVP:
    """绑定已有CudaPrimitiveContacts，重算当前接触并返回同方向几何导数。

    不拥有provider，不负责关闭它。nu=0退化情形明确拒绝普通导数声明。
    盒子支持sphere-box及capsule-box的已验证分支；plane-box仍拒绝。
    胶囊盒子近轴平行/内部/接触架与激活边界采用保守拒绝，不改变前向。
    """
    def __init__(self,provider):
        if not isinstance(provider,CudaPrimitiveContacts):raise TypeError('需要CudaPrimitiveContacts')
        if provider.closed:raise RuntimeError('碰撞provider已关闭')
        self.provider=provider;self.layout=provider.layout;self.numeric=Numerics('mixed64')
        types=provider.gmeta[:provider.ng];bodies=provider.gmeta[provider.ng:2*provider.ng]
        used=np.unique(provider.gmeta[2*provider.ng:])
        pairs=provider.gmeta[2*provider.ng:].reshape(-1,2)
        for a,b in pairs:
            if 3 in (types[a],types[b]) and (types[a],types[b]) not in ((1,3),(2,3)):
                raise ValueError('box几何反向支持sphere-box及保守守卫的capsule-box；plane-box尚未实现')
        for g in used[types[used]==0]:
            if bodies[g] not in (0,self.layout.palm_body_id):raise ValueError('原几何反向只支持world/规定手掌上的plane')
        self.has_capsule_box=any((types[a],types[b])==(2,3) for a,b in pairs)
        m=SimpleNamespace(nbody=provider.nb,nq=self.layout.nq,nv=self.layout.nv)
        poses=CONTACT_CUDA[:CONTACT_CUDA.index('__global__ void rows(')]
        prim=_collision_algorithm(include_boxes=True).split('template <class S>')[0]
        helpers='\n'.join(_original(n) for n in ('adj_kappa','adj_kappa_nu','make_frame_vjp','sphere_core_vjp','adj_contact_geom'))
        prefix=source.header(m,1,1)+f'\n#define NG {provider.ng}\n#define NP {provider.npair}\n'+source.DEVICE+poses+prim+PREAMBLE+helpers+BOX_VJP
        kernel=KERNEL
        device=''
        if self.has_capsule_box:
            device=capsule_box_source().split('__global__ void kernel(')[0]+CAPSULE_HELPER
            kernel=_patch_kernel(kernel,' double dist,pos[3]',CAPSULE_DISPATCH+' double dist,pos[3]')
            kernel=_patch_kernel(kernel,'GC(cudaMalloc(&db,2*nc*sizeof(int)))','GC(cudaMalloc(&db,(2*nc+NP)*sizeof(int)))')
            kernel=_patch_kernel(kernel,'nc*13*sizeof(double)','(nc?nc:1)*13*sizeof(double)',2)
            kernel=_patch_kernel(kernel,'nc*3*sizeof(int)','(nc?nc:1)*3*sizeof(int)')
            kernel=_patch_kernel(kernel,'geometry_jvp<<<', 'capsule_guards<<<(NP+63)/64,64>>>(dg,df,dp,db+2*nc);if(nc)geometry_jvp<<<')
            kernel=_patch_kernel(kernel,'GC(cudaMemcpy(branches,db,2*nc*sizeof(int)', 'GC(cudaMemcpy(branches,db,(2*nc+NP)*sizeof(int)')
        with tempfile.TemporaryDirectory(prefix='nabla_collision_adj_') as tmp:
            cu,so=Path(tmp)/'geometry.cu',Path(tmp)/'geometry.so'
            cu.write_text('#include <float.h>\n'+self.numeric.cuda(prefix)+device+self.numeric.cuda(kernel))
            compile_cuda(cu,so,extra=('-shared','-Xcompiler','-fPIC'));self._library=ctypes.CDLL(str(so))
        self._run=self._library.run;dp=self.numeric.pointer;ip=ctypes.POINTER(ctypes.c_int)
        self._run.argtypes=[ctypes.c_int,ip,ip,dp,dp,ip,dp,dp,dp,ip];self._run.restype=ctypes.c_int

    def evaluate(self,state,palm,direction):
        d=self.numeric.finite(direction,'direction')
        if d.shape!=(self.layout.nv,):raise ValueError('direction长度必须为nv')
        contacts,pairs,slots=self.provider.evaluate_with_slots(state,palm);nc=len(contacts)
        replay=np.empty((nc,13));out=np.empty((nc,13));branch_storage=np.empty(2*nc+(self.provider.npair if self.has_capsule_box else 0),np.int32)
        # 尾部每pair一项守卫，包括未生成接触行的胶囊激活边界。
        br=branch_storage[:2*nc].reshape(nc,2)
        ids=np.asarray(pairs,dtype=np.int32);ss=np.asarray(slots,dtype=np.int32)
        if nc or self.has_capsule_box:
            p=self.provider
            active=np.ascontiguousarray(np.c_[ids,ss,p.reverse[ids]],dtype=np.int32)
            exact=self.numeric.finite(np.r_[state.qpos,palm.position,palm.quaternion,p.geometry],'几何输入')
            dp=self.numeric.pointer;ip=ctypes.POINTER(ctypes.c_int)
            status=self._run(nc,active.ctypes.data_as(ip),p.gmeta.ctypes.data_as(ip),p.fixed.ctypes.data_as(dp),exact.ctypes.data_as(dp),p.meta.ctypes.data_as(ip),d.ctypes.data_as(dp),replay.ctypes.data_as(dp),out.ctypes.data_as(dp),branch_storage.ctypes.data_as(ip))
            if status:raise RuntimeError(f'CUDA碰撞几何导数失败:{status}')
            if self.has_capsule_box and np.any(branch_storage[2*nc:]):
                flags=int(np.bitwise_or.reduce(branch_storage[2*nc:]))
                names=[name for bit,name in ((1,'轴平行/近轴平行'),(2,'接触架切换'),(4,'内部/近零球盒距离'),(8,'接触激活边界')) if flags&bit]
                raise ValueError('capsule-box几何导数保守拒绝：'+','.join(names))
            if np.any(br[:,0]==-2):raise ValueError('sphere-box位于形状特征/最近面/接触架切换边界：不声明唯一经典导数')
            if status or not np.isfinite(out).all() or np.any(br[:,0]<0):raise RuntimeError(f'CUDA碰撞几何导数失败:{status}')
            if np.any(br[:,1]):raise ValueError('碰撞中心/轴距离为零：不声明唯一经典几何导数')
            expected=np.array([np.r_[c.distance,c.position,c.frame.ravel()] for c in contacts]);error=float(np.max(abs(replay-expected))) if nc else 0.
            if error>2e-11:raise RuntimeError(f'几何反向重放与前向不一致:{error}')
        else:error=0.
        for a in (ids,ss,out,br):a.setflags(write=False)
        return CollisionGeometryJVP(tuple(contacts),ids,ss,br[:,0],out[:,0],out[:,1:4],out[:,4:].reshape(nc,3,3),error)
