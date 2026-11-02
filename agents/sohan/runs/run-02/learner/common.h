#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#define W 19
#define IW 20
typedef struct { int nr; int x[4],y[4],w[4],h[4],wt[4]; } Feat;
typedef struct { Feat f; float thr; int par; float alpha; } Weak;
typedef struct { int n; Weak *w; float theta, sumA; } Stage;
static int read_pgm(const char*p, unsigned char*o){
  FILE*f=fopen(p,"rb"); if(!f) return 0; int w,h,m;
  if(fscanf(f,"P5 %d %d %d",&w,&h,&m)!=3||w!=W||h!=W){fclose(f);return 0;}
  fgetc(f); int r=fread(o,1,W*W,f); fclose(f); return r==W*W; }
/* integral image + normalization std */
static float make_ii(const unsigned char*im,int*ii){
  double s=0,s2=0; memset(ii,0,sizeof(int)*IW*IW);
  for(int y=0;y<W;y++){int row=0;for(int x=0;x<W;x++){int p=im[y*W+x];row+=p;s+=p;s2+=(double)p*p;
    ii[(y+1)*IW+x+1]=ii[y*IW+x+1]+row;}}
  double n=W*W,m=s/n,v=s2/n-m*m; float sd=sqrt(v>0?v:0); return sd<1?1:sd; }
static inline int rsum(const int*ii,int x,int y,int w,int h){
  return ii[(y+h)*IW+x+w]-ii[y*IW+x+w]-ii[(y+h)*IW+x]+ii[y*IW+x]; }
static inline float feval(const Feat*f,const int*ii,float sd){
  int s=0; for(int i=0;i<f->nr;i++) s+=f->wt[i]*rsum(ii,f->x[i],f->y[i],f->w[i],f->h[i]);
  return s/sd; }
static inline float weak_h(const Weak*w,const int*ii,float sd){
  float v=feval(&w->f,ii,sd); int face = w->par>0 ? v>w->thr : v<=w->thr; return face?1.f:-1.f; }
static float stage_sum(const Stage*s,const int*ii,float sd){
  float t=0; for(int i=0;i<s->n;i++) t+=s->w[i].alpha*weak_h(&s->w[i],ii,sd); return t; }
