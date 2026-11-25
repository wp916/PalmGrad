"""仅接触投影使用的双精度位姿/力臂运算；不改RNE、质量矩阵、软模型或积分。"""
CUDA=r'''
__device__ void cp_cross(const double* a,const double* b,double* c){
  c[0]=a[1]*b[2]-a[2]*b[1];c[1]=a[2]*b[0]-a[0]*b[2];c[2]=a[0]*b[1]-a[1]*b[0];
}
__device__ double cp_dot(const double* a,const double* b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
__device__ void cp_qmul(const double* a,const double* b,double* c){
  c[0]=a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3];
  c[1]=a[0]*b[1]+a[1]*b[0]+a[2]*b[3]-a[3]*b[2];
  c[2]=a[0]*b[2]-a[1]*b[3]+a[2]*b[0]+a[3]*b[1];
  c[3]=a[0]*b[3]+a[1]*b[2]-a[2]*b[1]+a[3]*b[0];
}
__device__ void cp_qrot(const double* q,const double* v,double* o){
  double t[3],u[3];cp_cross(q+1,v,t);for(int k=0;k<3;k++)t[k]*=2.;cp_cross(q+1,t,u);
  for(int k=0;k<3;k++)o[k]=v[k]+q[0]*t[k]+u[k];
}
__global__ void cp_poses(const double* input,double* poses){
  const double* palm=input+NQ;const double* fixed=palm+7;
  for(int k=0;k<7*NB;k++)poses[k]=0.;poses[3]=1.;
  for(int b=1;b<NB;b++){
    double* dst=poses+7*b;int kind=jc_kind[b];
    if(kind==1){for(int k=0;k<7;k++)dst[k]=palm[k];}
    else if(kind==3){for(int k=0;k<7;k++)dst[k]=input[c_qadr[b]+k];}
    else if(kind==2){
      const double* parent=poses+7*c_parent[b];double offset[3],base[4],hinge[4];
      cp_qrot(parent+3,fixed+3*b,offset);
      for(int k=0;k<3;k++)dst[k]=parent[k]+offset[k];
      cp_qmul(parent+3,fixed+3*NB+4*b,base);
      double half=.5*input[c_qadr[b]],s=sin(half);hinge[0]=cos(half);
      for(int k=0;k<3;k++)hinge[k+1]=s*fixed[7*NB+3*b+k];
      cp_qmul(base,hinge,dst+3);
    }
  }
}
'''
