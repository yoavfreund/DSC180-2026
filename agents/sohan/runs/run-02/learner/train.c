#include "common.h"
#include <dirent.h>
#include <stdint.h>
#define MAXF 8000
#define NEGCAP 5000
#define MAXWEAK 200
#define MAXSTAGES 16
#define DR 0.995
#define FPSTOP 0.45
static unsigned long long rs=88172645463325252ULL;
static unsigned rnd(){ rs^=rs<<13; rs^=rs>>7; rs^=rs<<17; return (unsigned)(rs>>11); }
static double urand(){ return (rnd()&0xFFFFFF)/16777216.0; }
typedef unsigned char Img[W*W];
static Feat *feats; static int nf;
static void addf(Feat f){ if(nf%1==0){} feats[nf++]=f; }
static void enumerate(Feat**out,int*n){
  int cap=200000; Feat*all=malloc(sizeof(Feat)*cap); int c=0;
  for(int t=0;t<5;t++) for(int h=1;h<=W;h++) for(int w=1;w<=W;w++){
    int tw=(t==0||t==4)?2*w:(t==2?3*w:w), th=(t==1||t==4)?2*h:(t==3?3*h:h);
    if(tw>W||th>W) continue;
    for(int y=0;y+th<=W;y++) for(int x=0;x+tw<=W;x++){ Feat f; memset(&f,0,sizeof f);
      if(t==0){f.nr=2;f.x[0]=x;f.y[0]=y;f.wt[0]=-1;f.x[1]=x+w;f.y[1]=y;f.wt[1]=1;}
      else if(t==1){f.nr=2;f.x[0]=x;f.y[0]=y;f.wt[0]=-1;f.x[1]=x;f.y[1]=y+h;f.wt[1]=1;}
      else if(t==2){f.nr=3;for(int i=0;i<3;i++){f.x[i]=x+i*w;f.y[i]=y;f.wt[i]=(i==1)?-2:1;}}
      else if(t==3){f.nr=3;for(int i=0;i<3;i++){f.x[i]=x;f.y[i]=y+i*h;f.wt[i]=(i==1)?-2:1;}}
      else {f.nr=4;for(int i=0;i<4;i++){f.x[i]=x+(i&1)*w;f.y[i]=y+(i>>1)*h;f.wt[i]=((i&1)^(i>>1))?-1:1;}}
      for(int i=0;i<f.nr;i++){f.w[i]=w;f.h[i]=h;}
      if(c<cap) all[c++]=f; } }
  fprintf(stderr,"total features %d\n",c);
  if(c>MAXF){ for(int i=c-1;i>0;i--){int j=rnd()%(i+1);Feat t=all[i];all[i]=all[j];all[j]=t;} c=MAXF; }
  *out=all; *n=c; }
static void dihedral(const Img a,Img b,int k){
  for(int y=0;y<W;y++)for(int x=0;x<W;x++){ int u=x,v=y;
    if(k&1)u=W-1-u; if(k&2)v=W-1-v; if(k&4){int t=u;u=v;v=t;}
    b[y*W+x]=a[v*W+u]; } }
static void zoom(const Img a,Img b){
  double s=14+urand()*5; double ox=urand()*(W-s), oy=urand()*(W-s);
  for(int y=0;y<W;y++)for(int x=0;x<W;x++){ double fx=ox+(x+0.5)*s/W-0.5, fy=oy+(y+0.5)*s/W-0.5;
    int x0=(int)floor(fx),y0=(int)floor(fy); double ax=fx-x0,ay=fy-y0;
    if(x0<0){x0=0;ax=0;} if(y0<0){y0=0;ay=0;} if(x0>W-2){x0=W-2;ax=1;} if(y0>W-2){y0=W-2;ay=1;}
    double p=(1-ay)*((1-ax)*a[y0*W+x0]+ax*a[y0*W+x0+1])+ay*((1-ax)*a[(y0+1)*W+x0]+ax*a[(y0+1)*W+x0+1]);
    b[y*W+x]=(unsigned char)(p+0.5); } }
/* sample storage */
typedef struct { int ii[IW*IW]; float sd; } Sm;
static Sm *pos,*neg; static int np,nn;
static int passes(const Stage*S,int ns,const Sm*s){
  for(int k=0;k<ns;k++) if(stage_sum(&S[k],s->ii,s->sd)<S[k].theta) return 0; return 1; }
static int cmpd(const void*a,const void*b){double x=*(double*)a,y=*(double*)b;return x<y?-1:x>y;}
static int cmpf(const void*a,const void*b){ float x=((const float*)a)[0],y=((const float*)b)[0]; return x<y?-1:x>y; }
int main(int argc,char**argv){
  if(argc!=4) return 1;
  char path[1024]; FILE*lf=fopen(argv[2],"r"); if(!lf) return 1;
  int cap=20000; Img *pi=malloc(cap*sizeof(Img)),*ni=malloc(cap*sizeof(Img)); int npi=0,nni=0;
  char id[256]; int lab;
  while(fscanf(lf,"%255s %d",id,&lab)==2){ snprintf(path,sizeof path,"%s/%s.pgm",argv[1],id);
    Img im; if(!read_pgm(path,im)) continue; if(lab) memcpy(pi[npi++],im,sizeof im); else memcpy(ni[nni++],im,sizeof im); }
  fprintf(stderr,"loaded %d pos %d neg\n",npi,nni);
  enumerate(&feats,&nf);
  /* augmentation */
  pos=malloc(sizeof(Sm)*npi*2); np=0;
  for(int i=0;i<npi;i++){ Img b; for(int k=0;k<2;k++){ if(k) dihedral(pi[i],b,1); else memcpy(b,pi[i],sizeof b);
      pos[np].sd=make_ii(b,pos[np].ii); np++; } }
  int nmax=nni*20; neg=malloc(sizeof(Sm)*nmax); nn=0;
  for(int i=0;i<nni;i++){ Img b,c2;
    for(int k=0;k<8;k++){ dihedral(ni[i],b,k); for(int inv=0;inv<2;inv++){ memcpy(c2,b,sizeof b);
        if(inv) for(int j=0;j<W*W;j++) c2[j]=255-c2[j];
        neg[nn].sd=make_ii(c2,neg[nn].ii); nn++; } }
    for(int z=0;z<3;z++){ zoom(ni[i],b); if(rnd()&1) for(int j=0;j<W*W;j++) b[j]=255-b[j];
      neg[nn].sd=make_ii(b,neg[nn].ii); nn++; } }
  fprintf(stderr,"samples: pos %d neg pool %d\n",np,nn);
  Stage S[MAXSTAGES]; int ns=0;
  char *alive=malloc(nn); memset(alive,1,nn);
  float *sv=malloc((size_t)(np+NEGCAP)*MAXF*sizeof(float));
  uint16_t *si=malloc((size_t)(np+NEGCAP)*MAXF*sizeof(uint16_t));
  if(!sv||!si){fprintf(stderr,"oom\n");return 1;}
  int *sel=malloc(sizeof(int)*nn);
  while(ns<MAXSTAGES){
    int na=0; for(int i=0;i<nn;i++){ if(alive[i]&&passes(S,ns,&neg[i])) sel[na++]=i; else alive[i]=0; }
    fprintf(stderr,"stage %d: %d negatives alive\n",ns+1,na);
    if(na<150 && ns>=2) break;
    if(na<20) break;
    for(int i=na-1;i>0;i--){int j=rnd()%(i+1);int t=sel[i];sel[i]=sel[j];sel[j]=t;}
    int nneg=na<NEGCAP?na:NEGCAP; int N=np+nneg;
    /* feature values */
    float *tmp=malloc(sizeof(float)*2*N);
    for(int f=0;f<nf;f++){ for(int s=0;s<N;s++){ const Sm*m=s<np?&pos[s]:&neg[sel[s-np]];
        tmp[2*s]=feval(&feats[f],m->ii,m->sd); ((int*)tmp)[2*s+1]=s; }
      qsort(tmp,N,8,cmpf);
      for(int s=0;s<N;s++){ sv[(size_t)f*N+s]=tmp[2*s]; si[(size_t)f*N+s]=((int*)tmp)[2*s+1]; } }
    free(tmp);
    double *w=malloc(sizeof(double)*N); double *sc=calloc(N,sizeof(double));
    for(int s=0;s<N;s++) w[s]=s<np?0.5/np:0.5/nneg;
    Stage*cur=&S[ns]; cur->w=calloc(MAXWEAK,sizeof(Weak)); cur->n=0; cur->sumA=0;
    double *ps=malloc(sizeof(double)*np);
    while(cur->n<MAXWEAK){
      double tot=0; for(int s=0;s<N;s++) tot+=w[s]; for(int s=0;s<N;s++) w[s]/=tot;
      double Wn=0; for(int s=np;s<N;s++) Wn+=w[s];
      double best=2; int bf=-1,bpar=1; float bthr=0;
      for(int f=0;f<nf;f++){ const float*v=sv+(size_t)f*N; const uint16_t*ix=si+(size_t)f*N;
        double wpb=0,wnb=0; /* error for par=+: pos below + neg above */
        for(int k=0;k<N-1;k++){ int s=ix[k]; if(s<np) wpb+=w[s]; else wnb+=w[s];
          if(v[k]<v[k+1]){ double e=wpb+(Wn-wnb); int par=1; if(e>0.5){e=1-e;par=-1;}
            if(e<best){best=e;bf=f;bpar=par;bthr=0.5f*(v[k]+v[k+1]);} } } }
      if(bf<0) break;
      double e=best<1e-6?1e-6:best; double alpha=0.5*log((1-e)/e);
      Weak*wk=&cur->w[cur->n++]; wk->f=feats[bf]; wk->thr=bthr; wk->par=bpar; wk->alpha=alpha; cur->sumA+=alpha;
      for(int s=0;s<N;s++){ const Sm*m=s<np?&pos[s]:&neg[sel[s-np]]; float h=weak_h(wk,m->ii,m->sd);
        double y=s<np?1:-1; w[s]*=exp(-alpha*y*h); sc[s]+=alpha*h; }
      /* threshold */
      for(int s=0;s<np;s++) ps[s]=sc[s];
      qsort(ps,np,sizeof(double),cmpd);
      int idx=(int)((1-DR)*np); cur->theta=ps[idx]-1e-6;
      int fp=0; for(int s=np;s<N;s++) if(sc[s]>=cur->theta) fp++;
      double fpr=(double)fp/nneg;
      if(cur->n%10==0) fprintf(stderr,"  weak %d fpr %.3f\n",cur->n,fpr);
      if(fpr<=FPSTOP && cur->n>=3) break;
    }
    fprintf(stderr,"stage %d done: %d weak, theta %.3f\n",ns+1,cur->n,cur->theta);
    free(w);free(sc);free(ps); ns++;
  }
  FILE*o=fopen(argv[3],"w"); fprintf(o,"%d\n",ns);
  for(int k=0;k<ns;k++){ fprintf(o,"%d %.9g %.9g\n",S[k].n,S[k].theta,S[k].sumA);
    for(int i=0;i<S[k].n;i++){ Weak*w=&S[k].w[i]; fprintf(o,"%d",w->f.nr);
      for(int r=0;r<w->f.nr;r++) fprintf(o," %d %d %d %d %d",w->f.x[r],w->f.y[r],w->f.w[r],w->f.h[r],w->f.wt[r]);
      fprintf(o," %.9g %d %.9g\n",w->thr,w->par,w->alpha); } }
  fclose(o); return 0; }
