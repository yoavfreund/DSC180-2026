#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <dirent.h>
#define NF 6500
#define B 128
#define MAXW 320
#define MAXS 4
typedef struct {int x,y,w,h,t;} Feature;
typedef struct {Feature f; int cut,pol; double a;} Weak;
typedef struct {int n; double threshold; Weak w[MAXW];} Stage;
typedef struct {int magic,n; Stage s[MAXS];} Model;
typedef struct {float ii[400];float sd;int y;} Sample;
#ifdef TRAIN
static unsigned rng=1234567;
static unsigned rnd(void){rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;return rng;}
#endif
static void prep(Sample *s,unsigned char *p,int label){double sum=0,sq=0; memset(s,0,sizeof(*s));s->y=label;for(int y=0;y<19;y++)for(int x=0;x<19;x++){double v=p[y*19+x];sum+=v;sq+=v*v;s->ii[(y+1)*20+x+1]=v+s->ii[y*20+x+1]+s->ii[(y+1)*20+x]-s->ii[y*20+x];}s->sd=sqrt(fmax(1.,sq/361-sum*sum/(361.*361.)));}
static int readpgm(const char *path,unsigned char *p){FILE *f=fopen(path,"rb");if(!f)return 0;char head[64];int w,h,m;if(!fgets(head,sizeof(head),f)||strcmp(head,"P5\n")||fscanf(f,"%d%d%d",&w,&h,&m)!=3||w!=19||h!=19||m!=255){fclose(f);return 0;}fgetc(f);int ok=fread(p,1,361,f)==361;fclose(f);return ok;}
static float rect(const Sample*s,int x,int y,int w,int h){const float *p=s->ii;return p[(y+h)*20+x+w]-p[y*20+x+w]-p[(y+h)*20+x]+p[y*20+x];}
static int val(const Sample*s,Feature f){float z=0,a=rect(s,f.x,f.y,f.w,f.h);if(f.t==0)z=a-2*rect(s,f.x,f.y,f.w/2,f.h);if(f.t==1)z=a-2*rect(s,f.x,f.y,f.w,f.h/2);if(f.t==2)z=a-3*rect(s,f.x+f.w/3,f.y,f.w/3,f.h);if(f.t==3)z=a-3*rect(s,f.x,f.y+f.h/3,f.w,f.h/3);if(f.t==4)z=a-2*(rect(s,f.x,f.y,f.w/2,f.h/2)+rect(s,f.x+f.w/2,f.y+f.h/2,f.w/2,f.h/2));z=z/(f.w*f.h*s->sd);int b=(int)(64+z*24);return b<0?0:b>127?127:b;}
#ifndef TRAIN
static double response(const Sample*s,const Weak*w){return w->a*((val(s,w->f)<w->cut?1:-1)*w->pol);}
#endif
#ifdef TRAIN
static int cmp(const void*a,const void*b){double x=*(double*)a,y=*(double*)b;return (x>y)-(x<y);}
int main(int argc,char**argv){if(argc!=4)return 1;FILE*f=fopen(argv[2],"r");if(!f)return 1;Sample *ss=calloc(40000,sizeof(Sample));int n=0,np=0;char id[256],path[4096];int y;unsigned char p[361],q[361];while(fscanf(f,"%255s%d",id,&y)==2){snprintf(path,sizeof(path),"%s/%s.pgm",argv[1],id);if(!readpgm(path,p))return 2;int nv=y?2:6;for(int v=0;v<nv;v++){for(int yy=0;yy<19;yy++)for(int xx=0;xx<19;xx++){int x=xx,yy2=yy;if(v==1)x=18-x;if(v>=2){x+=(v==2)-(v==3);yy2+=(v==4)-(v==5);}if(x<0)x=0;if(x>18)x=18;if(yy2<0)yy2=0;if(yy2>18)yy2=18;q[yy*19+xx]=p[yy2*19+x];}prep(ss+n,q,y);n++;np+=y;}}fclose(f);
Feature *features=calloc(NF,sizeof(Feature));for(int j=0;j<NF;j++){Feature a; a.t=rnd()%5;int dx=(a.t==0||a.t==4)?2:a.t==2?3:1;int dy=(a.t==1||a.t==4)?2:a.t==3?3:1;a.w=dx*(1+rnd()%(19/dx));a.h=dy*(1+rnd()%(19/dy));a.x=rnd()%(20-a.w);a.y=rnd()%(20-a.h);features[j]=a;}
uint8_t *data=malloc((size_t)NF*n);for(int j=0;j<NF;j++)for(int i=0;i<n;i++)data[(size_t)j*n+i]=val(ss+i,features[j]);
int *active=malloc(n*sizeof(int));double *weights=malloc(n*sizeof(double)),*scores=calloc(n,sizeof(double)),*pos=malloc(np*sizeof(double));int na=n;for(int i=0;i<n;i++)active[i]=i;Model *m=calloc(1,sizeof(Model));m->magic=0x43415332;int sizes[]={8,24,48,80};
for(int st=0;st<MAXS;st++){int nn=0;for(int k=0;k<na;k++)nn+=!ss[active[k]].y;if(!nn&&st>=2)break;Stage *s=&m->s[st];m->n++;for(int k=0;k<na;k++){int i=active[k];weights[i]=ss[i].y?.5/np:.5/fmax(1,nn);scores[i]=0;}
for(int t=0;t<sizes[st];t++){double best=1e10;int bj=0,bc=0,bp=1;double tp=0,tn=0;for(int k=0;k<na;k++){int i=active[k];if(ss[i].y)tp+=weights[i];else tn+=weights[i];}
for(int j=0;j<NF;j++){double hp[B]={0},hn[B]={0};uint8_t*d=data+(size_t)j*n;for(int k=0;k<na;k++){int i=active[k];if(ss[i].y)hp[d[i]]+=weights[i];else hn[d[i]]+=weights[i];}double cp=0,cn=0;for(int b=0;b<=B;b++){double e=cn+tp-cp;int pol=1;if(tp+tn-e<e){e=tp+tn-e;pol=-1;}if(e<best){best=e;bj=j;bc=b;bp=pol;}if(b<B){cp+=hp[b];cn+=hn[b];}}}
double e=fmax(1e-9,fmin(.499999,best/(tp+tn)));Weak *w=&s->w[s->n++];w->f=features[bj];w->cut=bc;w->pol=bp;w->a=fmin(2.,.5*log((1-e)/e));double total=0;for(int k=0;k<na;k++){int i=active[k];double h=w->a*((data[(size_t)bj*n+i]<bc?1:-1)*bp);scores[i]+=h;weights[i]*=exp(ss[i].y?-h:h);total+=weights[i];}for(int k=0;k<na;k++)weights[active[k]]/=total;
}
int pn=0;for(int k=0;k<na;k++)if(ss[active[k]].y)pos[pn++]=scores[active[k]];qsort(pos,pn,sizeof(double),cmp);s->threshold=pos[(int)(.0015*pn)]-.25;int out=0,left=0;for(int k=0;k<na;k++){int i=active[k];if(ss[i].y||scores[i]>=s->threshold){active[out++]=i;left+=!ss[i].y;}}na=out;fprintf(stderr,"stage %d weak %d negatives %d -> %d threshold %.4f\n",st+1,s->n,nn,left,s->threshold);
}
f=fopen(argv[3],"wb");if(!f)return 3;fwrite(m,sizeof(*m),1,f);fclose(f);return 0;}
#else
int main(int argc,char**argv){if(argc!=3)return 1;int desc=!strcmp(argv[1],"--describe");FILE*f=fopen(argv[desc?2:1],"rb");if(!f)return 2;Model *m=malloc(sizeof(Model));int ok=fread(m,sizeof(*m),1,f)==1;fclose(f);if(!ok||m->magic!=0x43415332||m->n<2||m->n>MAXS)return 3;if(desc){for(int k=0;k<m->n;k++)printf("stage=%d weak=%d\n",k+1,m->s[k].n);return 0;}DIR*d=opendir(argv[2]);if(!d)return 4;struct dirent*ent;char path[4096];unsigned char p[361];while((ent=readdir(d))){size_t len=strlen(ent->d_name);if(len<5||strcmp(ent->d_name+len-4,".pgm"))continue;snprintf(path,sizeof(path),"%s/%s",argv[2],ent->d_name);if(!readpgm(path,p))return 5;Sample s;prep(&s,p,0);double margin=0;int k;for(k=0;k<m->n;k++){double z=0;for(int j=0;j<m->s[k].n;j++)z+=response(&s,&m->s[k].w[j]);margin=z-m->s[k].threshold;if(margin<0||k==m->n-1)break;}double sig=1/(1+exp(-fmax(-30,fmin(30,margin))));sig=fmax(1e-9,fmin(1-1e-9,sig));ent->d_name[len-4]=0;printf("%s %.12f\n",ent->d_name,k+sig);}closedir(d);return 0;}
#endif
