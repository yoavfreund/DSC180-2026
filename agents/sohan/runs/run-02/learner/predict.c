#include "common.h"
#include <dirent.h>
static Stage*st; static int ns;
static int load(const char*p){
  FILE*f=fopen(p,"r"); if(!f) return 0; if(fscanf(f,"%d",&ns)!=1) return 0;
  st=calloc(ns,sizeof(Stage));
  for(int k=0;k<ns;k++){ if(fscanf(f,"%d %f %f",&st[k].n,&st[k].theta,&st[k].sumA)!=3) return 0;
    st[k].w=calloc(st[k].n,sizeof(Weak));
    for(int i=0;i<st[k].n;i++){ Weak*w=&st[k].w[i]; if(fscanf(f,"%d",&w->f.nr)!=1) return 0;
      for(int r=0;r<w->f.nr;r++) if(fscanf(f,"%d %d %d %d %d",&w->f.x[r],&w->f.y[r],&w->f.w[r],&w->f.h[r],&w->f.wt[r])!=5) return 0;
      if(fscanf(f,"%f %d %f",&w->thr,&w->par,&w->alpha)!=3) return 0; } }
  fclose(f); return 1; }
int main(int c,char**v){
  if(c==3&&!strcmp(v[1],"--describe")){ if(!load(v[2])) return 1;
    for(int k=0;k<ns;k++) printf("stage=%d weak=%d\n",k+1,st[k].n); return 0; }
  if(c!=3) return 1; if(!load(v[1])) return 1;
  DIR*d=opendir(v[2]); if(!d) return 1; struct dirent*e; char path[1024];
  while((e=readdir(d))){ size_t l=strlen(e->d_name); if(l<5||strcmp(e->d_name+l-4,".pgm")) continue;
    snprintf(path,sizeof path,"%s/%s",v[2],e->d_name); unsigned char im[W*W]; int ii[IW*IW];
    if(!read_pgm(path,im)) continue; float sd=make_ii(im,ii); int k; double m=0;
    for(k=0;k<ns;k++){ m=stage_sum(&st[k],ii,sd)-st[k].theta; if(m<0) break; }
    double z=4*m/(st[k<ns?k:ns-1].sumA+1e-9); double sc=k+1/(1+exp(-z));
    printf("%.*s %.6f\n",(int)(l-4),e->d_name,sc); }
  closedir(d); return 0; }
