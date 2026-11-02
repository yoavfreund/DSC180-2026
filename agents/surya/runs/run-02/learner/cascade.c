/* Fixed Haar feature pool and discrete AdaBoost. Search is capped by the precommitted plan. */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <dirent.h>
#define F 6500
#define MAX 40000
typedef struct{
    int t,x,y,w,h;
}
Feature;
typedef struct{
    int f,b,p;
    double a;
}
Weak;
typedef struct{
    int n;
    double th;
    Weak w[64];
}
Stage;
typedef struct{
    int n;
    Stage s[8];
}
Model;
typedef struct{
    char id[128];
    int y,part;
    unsigned char pix[361];
}
Sample;
FILE *detail;
Feature feat[F];
Sample *samples;
unsigned char *cache;
int N;
uint32_t rng=1234567;
uint32_t draw(void){
    rng^=rng<<13;
    rng^=rng>>17;
    rng^=rng<<5;
    return rng;
}
/* Data-independent xorshift feature recipe. */
void features(void){
    for(int i=0;i<F;i++){
        Feature *f=feat+i;
        f->t=draw()%5;
        int dx=(f->t==0||f->t==4)?2:f->t==2?3:1,dy=(f->t==1||f->t==4)?2:f->t==3?3:1;
        f->w=dx*(1+draw()%(19/dx));
        f->h=dy*(1+draw()%(19/dy));
        f->x=draw()%(20-f->w);
        f->y=draw()%(20-f->h);
    }
}
int rect(int *ii,int x,int y,int w,int h){
    return ii[(y+h)*20+x+w]-ii[y*20+x+w]-ii[(y+h)*20+x]+ii[y*20+x];
}
double integral(unsigned char *pix,int *ii){
    memset(ii,0,400*sizeof(int));
    double sum=0,sq=0;
    for(int y=0;y<19;y++)for(int x=0;x<19;x++){
        int v=pix[y*19+x];
        sum+=v;
        sq+=v*v;
        ii[(y+1)*20+x+1]=v+ii[y*20+x+1]+ii[(y+1)*20+x]-ii[y*20+x];
    }
    return sqrt(fmax(1,sq/361-(sum/361)*(sum/361)));
}
/* Shared quantization for training and lazy prediction. */
int quantized(int index,int *ii,double sd){
    Feature f=feat[index];
    int x=f.x,y=f.y,w=f.w,h=f.h;
    double v=rect(ii,x,y,w,h);
    if(f.t==0)v-=2*rect(ii,x,y,w/2,h);
    if(f.t==1)v-=2*rect(ii,x,y,w,h/2);
    if(f.t==2)v-=3*rect(ii,x+w/3,y,w/3,h);
    if(f.t==3)v-=3*rect(ii,x,y+h/3,w,h/3);
    if(f.t==4)v-=2*(rect(ii,x,y,w/2,h/2)+rect(ii,x+w/2,y+h/2,w/2,h/2));
    int q=(int)(64+24*v/(w*h*sd));
    return q<0?0:q>127?127:q;
}
void response(unsigned char *pix,unsigned char *out){
    int ii[400];
    double sd=integral(pix,ii);
    for(int i=0;i<F;i++)out[i]=quantized(i,ii,sd);
}
void token(FILE*f,char*b){
    int c;
    do{
        c=fgetc(f);
        if(c=='#')while(c!='\n'&&c!=EOF)c=fgetc(f);
    }
    while(c<=32&&c!=EOF);
    int n=0;
    while(c>32&&c!=EOF){
        b[n++]=c;
        c=fgetc(f);
    }
    b[n]=0;
}
void pgm(char*path,unsigned char*p){
    FILE*f=fopen(path,"rb");
    if(!f){
        perror(path);
        exit(1);
    }
    char b[128];
    token(f,b);
    if(strcmp(b,"P5")){
        fprintf(stderr,"Not P5\n");
        exit(1);
    }
    token(f,b);
    if(atoi(b)!=19)exit(1);
    token(f,b);
    if(atoi(b)!=19)exit(1);
    token(f,b);
    if(atoi(b)!=255)exit(1);
    if(fread(p,1,361,f)!=361)exit(1);
    fclose(f);
}
int member(char *id,char ids[][128],int n){
    for(int i=0;i<n;i++)if(!strcmp(id,ids[i]))return 1;
    return 0;
}
int readids(char*path,char ids[][128]){
    FILE*f=fopen(path,"r");
    if(!f)exit(1);
    int n=0,y;
    while(fscanf(f,"%127s%d",ids[n],&y)==2)n++;
    fclose(f);
    return n;
}
void data(char*dir,char*labels,int final){
    static char ci[6000][128],si[6000][128];
    int nc=readids("calibration.txt",ci),ns=readids("selection.txt",si);
    samples=calloc(MAX,sizeof(Sample));
    FILE*f=fopen(labels,"r");
    char id[128],path[1024];
    int y;
    while(fscanf(f,"%127s%d",id,&y)==2){
        int part=member(id,ci,nc)?1:(!final&&member(id,si,ns)?2:0);
        Sample orig={
            .y=y,.part=part
        }
        ;
        strcpy(orig.id,id);
        snprintf(path,sizeof path,"%s/%s%s",dir,id,strstr(id,".pgm")?"":".pgm");
        pgm(path,orig.pix);
        int count=part?1:(y?2:6);
        for(int a=0;a<count;a++){
            Sample *s=&samples[N++];
            *s=orig;
            for(int yy=0;yy<19;yy++)for(int xx=0;xx<19;xx++){
                int x=xx,z=yy;
                if(a==1)x=18-xx;
                if(a==2)x=xx-1;
                if(a==3)x=xx+1;
                if(a==4)z=yy-1;
                if(a==5)z=yy+1;
                x=x<0?0:x>18?18:x;
                z=z<0?0:z>18?18:z;
                s->pix[yy*19+xx]=orig.pix[z*19+x];
            }
        }
    }
    fclose(f);
    cache=malloc((size_t)F*N);
    unsigned char vals[F];
    for(int i=0;i<N;i++){
        response(samples[i].pix,vals);
        for(int j=0;j<F;j++)cache[(size_t)j*N+i]=vals[j];
    }
    fprintf(stderr,"loaded %d examples\n",N);
}
double value(Stage*s,int i){
    double v=0;
    for(int j=0;j<s->n;j++){
        Weak w=s->w[j];
        v+=w.a*w.p*(cache[(size_t)w.f*N+i]<=w.b?1:-1);
    }
    return v;
}
int cmpd(const void*a,const void*b){
    double x=*(double*)a,y=*(double*)b;
    return (x>y)-(x<y);
}
double threshold(double*v,int n,double retention,double offset){
    if(!n)return -1e30;
    qsort(v,n,sizeof(double),cmpd);
    int k=(int)floor((1-retention)*n+1e-9);
    return v[k]-offset;
}
void save(char*path,Model*m){
    FILE*f=fopen(path,"w");
    fprintf(f,"CASCADE1 %d\n",m->n);
    for(int k=0;k<m->n;k++){
        Stage*s=m->s+k;
        fprintf(f,"%d %.17g\n",s->n,s->th);
        for(int j=0;j<s->n;j++){
            Weak w=s->w[j];
            fprintf(f,"%d %d %d %.17g\n",w.f,w.b,w.p,w.a);
        }
    }
    fclose(f);
}
void load(char*path,Model*m){
    FILE*f=fopen(path,"r");
    if(!f)exit(1);
    char b[32];
    if(fscanf(f,"%31s%d",b,&m->n)!=2||strcmp(b,"CASCADE1")||m->n<2||m->n>8)exit(1);
    for(int k=0;k<m->n;k++){
        Stage*s=m->s+k;
        if(fscanf(f,"%d%lf",&s->n,&s->th)!=2||s->n<1||s->n>64||!isfinite(s->th))exit(1);
        for(int j=0;j<s->n;j++){
            Weak*w=s->w+j;
            if(fscanf(f,"%d%d%d%lf",&w->f,&w->b,&w->p,&w->a)!=4||w->f<0||w->f>=F||!isfinite(w->a))exit(1);
        }
    }
    fclose(f);
}
/* Build a candidate or retrain frozen stage sizes from fresh weights. */
void fit(Model*m,int fixed,int *sizes,double target,double retention,double offset){
    int *alive=malloc(N*sizeof(int));
    double *wt=calloc(N,sizeof(double)),*scores=calloc(N,sizeof(double)),*vals=malloc(N*sizeof(double));
    int total=0,baseNeg=0;
    for(int i=0;i<N;i++){
        alive[i]=1;
        if(samples[i].part==1&&!samples[i].y)baseNeg++;
    }
    for(int k=0;k<(fixed?fixed:8)&&total+2<=160;k++){
        Stage*s=&m->s[k];
        int pos=0,neg=0,cn=0,cp=0;
        for(int i=0;i<N;i++){
            scores[i]=0;
            if(!samples[i].part&&(samples[i].y||alive[i])){
                if(samples[i].y)pos++;
                else neg++;
            }
            if(alive[i]&&samples[i].part==1){
                if(samples[i].y)cp++;
                else cn++;
            }
        }
        for(int i=0;i<N;i++)wt[i]=(!samples[i].part&&(samples[i].y||alive[i]))?.5/(samples[i].y?pos:(neg?neg:1)):0;
        int cap=fixed?sizes[k]:64;
        if(cap>160-total)cap=160-total;
        double pass=1;
        for(int j=0;j<cap;j++){
            Weak best={
                0,0,1,0
            }
            ;
            double err=1;
            if(neg&&pos){
                double tp=0,tn=0;
                for(int i=0;i<N;i++)if(samples[i].y)tp+=wt[i];
                else tn+=wt[i];
                for(int f=0;f<F;f++){
                    double hp[128]={
                        0
                    }
                    ,hn[128]={
                        0
                    }
                    ;
                    unsigned char *c=cache+(size_t)f*N;
                    for(int i=0;i<N;i++)if(wt[i]){
                        if(samples[i].y)hp[c[i]]+=wt[i];
                        else hn[c[i]]+=wt[i];
                    }
                    double lp=0,ln=0;
                    for(int b=0;b<128;b++){
                        lp+=hp[b];
                        ln+=hn[b];
                        double e=ln+tp-lp;
                        if(e<err){
                            err=e;
                            best=(Weak){
                                f,b,1,0
                            }
                            ;
                        }
                        e=lp+tn-ln;
                        if(e<err){
                            err=e;
                            best=(Weak){
                                f,b,-1,0
                            }
                            ;
                        }
                    }
                }
                err=fmax(1e-12,fmin(1-1e-12,err));
                best.a=fmin(2,.5*log((1-err)/err));
            }
            s->w[s->n++]=best;
            total++;
            double norm=0;
            for(int i=0;i<N;i++){
                int h=best.p*(cache[(size_t)best.f*N+i]<=best.b?1:-1);
                scores[i]+=best.a*h;
                wt[i]*=exp(-best.a*(samples[i].y?1:-1)*h);
                norm+=wt[i];
            }
            if(norm)for(int i=0;i<N;i++)wt[i]/=norm;
            int nv=0;
            for(int i=0;i<N;i++)if(alive[i]&&samples[i].part==1&&samples[i].y)vals[nv++]=scores[i];
            s->th=threshold(vals,nv,retention,offset);
            int pn=0;
            for(int i=0;i<N;i++)if(alive[i]&&samples[i].part==1&&!samples[i].y&&scores[i]>=s->th)pn++;
            pass=cn?(double)pn/cn:0;
            if(!fixed&&s->n>=2&&pass<=target)break;
        }
        int remain=0;
        for(int i=0;i<N;i++){
            if(alive[i]&&scores[i]<s->th)alive[i]=0;
            if(alive[i]&&samples[i].part==1&&!samples[i].y)remain++;
        }
        m->n++;
        if(fixed)fprintf(stderr,"frozen stage %d weak %d cal_faces %d cal_neg %d pass %.9f remaining %d threshold %.9g\n",k+1,s->n,cp,cn,pass,remain,s->th);
        else fprintf(stderr,"stage %d weak %d cal_faces %d cal_neg %d pass %.9f target %.3f met %d remaining %d threshold %.9g\n",k+1,s->n,cp,cn,pass,target,pass<=target,remain,s->th);
        if(!fixed&&m->n>=2&&(!baseNeg||(double)remain/baseNeg<=.005))break;
    }
    free(alive);
    free(wt);
    free(scores);
    free(vals);
}
/* Sequential thresholds on independent, arriving calibration faces only. */
void calibrate(Model*m,double r,double o){
    int *a=malloc(N*sizeof(int));
    double*v=malloc(N*sizeof(double));
    for(int i=0;i<N;i++)a[i]=1;
    if(detail)fprintf(detail,"[\n");
    for(int k=0;k<m->n;k++){
        Stage*s=m->s+k;
        int n=0,cn=0;
        for(int i=0;i<N;i++)if(a[i]&&samples[i].part==1){
            if(samples[i].y)v[n++]=value(s,i);
            else cn++;
        }
        s->th=threshold(v,n,r,o);
        int fp=0,np=0;
        for(int i=0;i<N;i++){
            if(a[i]&&value(s,i)<s->th)a[i]=0;
            if(a[i]&&samples[i].part==1){
                if(samples[i].y)fp++;
                else np++;
            }
        }
        if(detail)fprintf(detail,"%s{\"stage\":%d,\"weak\":%d,\"threshold\":%.17g,\"arriving_faces\":%d,\"arriving_nonfaces\":%d,\"passing_faces\":%d,\"passing_nonfaces\":%d}",k?",\n":"",k+1,s->n,s->th,n,cn,fp,np);
    }
    if(detail)fprintf(detail,"\n]\n");
    free(a);
    free(v);
}
/* Internal holdout scores and actual evaluated weak counts. */
void scored(Model*m){
    for(int i=0;i<N;i++)if(samples[i].part){
        int ev=0,k;
        double margin=0;
        for(k=0;k<m->n;k++){
            ev+=m->s[k].n;
            margin=value(&m->s[k],i)-m->s[k].th;
            if(margin<0||k==m->n-1)break;
        }
        double z=1/(1+exp(-fmax(-700,fmin(700,margin))));
        z=fmax(1e-9,fmin(1-1e-9,z));
        printf("%s %d %d %.12f %d %d\n",samples[i].id,samples[i].part,samples[i].y,k+z,ev,k+1);
    }
}
int main(int argc,char**argv){
    features();
    Model m={
        0
    }
    ;
    if(strstr(argv[0],"predict")){
        if(argc==3&&!strcmp(argv[1],"--describe")){
            load(argv[2],&m);
            for(int k=0;k<m.n;k++)printf("stage=%d weak=%d\n",k+1,m.s[k].n);
            return 0;
        }
        if(argc!=3)return 1;
        load(argv[1],&m);
        DIR*d=opendir(argv[2]);
        struct dirent*e;
        unsigned char pix[361];
        int v[F],ii[400];
        long count=0,eval=0;
        while((e=readdir(d))){
            int len=strlen(e->d_name);
            if(len<4||strcmp(e->d_name+len-4,".pgm"))continue;
            char path[1024];
            snprintf(path,sizeof path,"%s/%s",argv[2],e->d_name);
            pgm(path,pix);
            double sd=integral(pix,ii);
            for(int j=0;j<F;j++)v[j]=-1;
            int k;
            double margin=0;
            for(k=0;k<m.n;k++){
                Stage*s=m.s+k;
                double score=0;
                for(int j=0;j<s->n;j++){
                    Weak w=s->w[j];
                    if(v[w.f]<0)v[w.f]=quantized(w.f,ii,sd);
                    score+=w.a*w.p*(v[w.f]<=w.b?1:-1);
                    eval++;
                }
                margin=score-s->th;
                if(margin<0||k==m.n-1)break;
            }
            double z=fmax(1e-9,fmin(1-1e-9,1/(1+exp(-fmax(-700,fmin(700,margin))))));
            e->d_name[len-4]=0;
            printf("%s %.12f\n",e->d_name,k+z);
            count++;
        }
        closedir(d);
        fprintf(stderr,"images=%ld evaluated_weak=%ld mean_weak=%.9f\n",count,eval,count?(double)eval/count:0);
        return 0;
    }
    if(argc>1&&!strcmp(argv[1],"--build")){
        data(argv[2],argv[3],0);
        FILE*f=fopen("feature_pool.tsv","w");
        for(int i=0;i<F;i++)fprintf(f,"%d %d %d %d %d\n",feat[i].t,feat[i].x,feat[i].y,feat[i].w,feat[i].h);
        fclose(f);
        fit(&m,0,NULL,atof(argv[5]),.998,0);
        save(argv[4],&m);
        return 0;
    }
    if(argc>1&&!strcmp(argv[1],"--sweep")){
        data(argv[2],argv[3],0);
        load(argv[4],&m);
        char dp[512];
        snprintf(dp,sizeof dp,"%s.r%s.o%s.json",argv[4],argv[5],argv[6]);
        detail=fopen(dp,"w");
        calibrate(&m,atof(argv[5]),atof(argv[6]));
        fclose(detail);
        detail=NULL;
        scored(&m);
        return 0;
    }
    if(argc!=4){
        return 1;
    }
    FILE*f=fopen("selected_config.json","r");
    if(!f){
        perror("selected_config.json");
        return 1;
    }
    char buf[4096];
    size_t n=fread(buf,1,sizeof(buf)-1,f);
    buf[n]=0;
    fclose(f);
    char*p=strstr(buf,"\"stage_sizes\"");
    if(!p)return 1;
    p=strchr(p,'[')+1;
    int sizes[8],ns=0;
    while(*p&&*p!=']'){
        if(*p>='0'&&*p<='9'){
            if(ns>=8)return 1;
            sizes[ns]=strtol(p,&p,10);
            if(sizes[ns]<2||sizes[ns]>64)return 1;
            ns++;
        }
        else p++;
    }
    double r=atof(strchr(strstr(buf,"\"retention\""),':')+1),o=atof(strchr(strstr(buf,"\"offset\""),':')+1);
    if(ns<2)return 1;
    int sum=0;for(int j=0;j<ns;j++)sum+=sizes[j];
    if(sum>160||!isfinite(r)||!isfinite(o))return 1;
    data(argv[1],argv[2],1);
    fit(&m,ns,sizes,-1,r,o);
    save(argv[3],&m);
    return 0;
}
