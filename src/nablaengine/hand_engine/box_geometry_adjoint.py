"""球—盒子固定特征分支的解析CUDA VJP。尺寸固定；余切输出为世界平移/旋转力矩。

前向使用原sphere_box重放；反向对盒子局部clamp/normalize和刚体变换求导。
最近面并列、clamp边界及接触架基选择边界不声明唯一普通导数。
"""

CUDA=r'''
__device__ int sphere_box_vjp(const double* sp,const double* bp,const double* bq,
                             double radius,const double* half,double* dist,double* pos,double* fr,
                             double* nu,const double* cot,double* gb){
 double R[9],d[3],c[3],cl[3],v[3],nl[3],nw[3];
 quat2mat(bq,R);sphere_box(sp,radius,bp,R,half,dist,pos,nw);make_frame(nw,fr);
 *nu=1.;for(int k=0;k<12;k++)gb[k]=0.;
 for(int k=0;k<3;k++)d[k]=sp[k]-bp[k];mtv3(R,d,c);
 double scale=fmax(half[0],fmax(half[1],half[2])),tol=16*DBL_EPSILON*scale;
 int branch=0;
 for(int k=0;k<3;k++){
  if(fabs(fabs(c[k])-half[k])<=tol)return -2;
  cl[k]=fmax(-half[k],fmin(half[k],c[k]));v[k]=cl[k]-c[k];
  if(-half[k]<c[k]&&c[k]<half[k])branch|=1<<(5+k);
 }
 double dd=sqrt(dot3(v,v));*nu=dd>1e-15?dd:1.;
 if(dd>0&&dd<=1e-15)return -2; // 原前向近零距离回退带，不声明内部点公式的导数。
 if(fabs(fabs(nw[1])-.5)<=16*DBL_EPSILON)return -2;
 if(!(-.5<nw[1]&&nw[1]<.5))branch|=ADJ_BR_FRAME_Z;
 double pbar[3],nb[3],nlbar[3],cb[3]={},pl[3];
 mtv3(R,cot+1,pbar);make_frame_vjp(fr,cot+4,nb);mtv3(R,nb,nlbar);
 if(dd>1e-15){
  for(int k=0;k<3;k++){nl[k]=v[k]/dd;pl[k]=.5*(cl[k]+c[k]+radius*nl[k]);nlbar[k]+=.5*radius*pbar[k];}
  double normalpart=dot3(nl,nlbar);
  for(int k=0;k<3;k++){
   double vb=nl[k]*cot[0]+(nlbar[k]-nl[k]*normalpart)/dd;
   cb[k]=.5*pbar[k]-vb;
   if(branch&(1<<(5+k)))cb[k]+=.5*pbar[k]+vb;
  }
 }else{
  // 原sphere_box的内部点最近面选择及方向；严格并列显式拒绝。
  double closest=2*(half[0]+half[1]+half[2]);int face=0;
  for(int i=0;i<6;i++){
   double fd=fabs(((i%2)?1.:-1.)*half[i/2]-c[i/2]);
   if(closest>fd){closest=fd;face=i;}
  }
  for(int i=0;i<6;i++)if(i!=face){
   double fd=fabs(((i%2)?1.:-1.)*half[i/2]-c[i/2]);if(fabs(fd-closest)<=tol)return -2;
  }
  for(int k=0;k<3;k++)nl[k]=0.;nl[face/2]=(face%2)?-1.:1.;
  double closestbar=-cot[0]-.5*dot3(pbar,nl);
  for(int k=0;k<3;k++){pl[k]=c[k]+nl[k]*(radius-closest)*.5;cb[k]=pbar[k]+closestbar*nl[k];}
  branch|=(1<<8)|(face<<9);
 }
 double worldcb[3],worldpl[3],a[3],b[3],e[3];mv3(R,cb,worldcb);mv3(R,pl,worldpl);
 cross(worldpl,cot+1,a);cross(nw,nb,b);cross(d,worldcb,e);
 for(int k=0;k<3;k++){gb[k]=worldcb[k];gb[6+k]=cot[1+k]-worldcb[k];gb[9+k]=a[k]+b[k]-e[k];}
 return branch;
}
__device__ int hand_contact_geom(int t1,int t2,int slot,const double* p1,const double* q1,
                                const double* s1,const double* p2,const double* q2,const double* s2,
                                double* dist,double* pos,double* fr,double* nu,const double* cot,double* gb){
 if(t1==1&&t2==3)return sphere_box_vjp(p1,p2,q2,s1[0],s2,dist,pos,fr,nu,cot,gb);
 return adj_contact_geom(t1,t2,slot,p1,q1,s1,p2,q2,s2,dist,pos,fr,nu,cot,gb);
}
'''
