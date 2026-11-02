import os
if os.path.exists("selected_config.json") or os.path.exists("construction_A.log"):
 raise SystemExit("Search construction budget already consumed; do not rerun.")
import subprocess,json,math,itertools,time
log={'constructions':[],'sweeps':[]}
def run(args):
 t=time.monotonic();p=subprocess.run(args,text=True,capture_output=True,check=True);return p,time.monotonic()-t
def layout(path):
 tok=open(path).read().split();i=2;st=[]
 for _ in range(int(tok[1])):
  n=int(tok[i]);th=float(tok[i+1]);st.append({'weak':n,'threshold':th});i+=2+4*n
 return st
def metrics(rows,part):
 rr=[r for r in rows if r[1]==part];np=sum(r[2] for r in rr);nn=len(rr)-np
 out={'faces':np,'nonfaces':nn,'mean_weak_face':sum(r[4] for r in rr if r[2])/np,'mean_weak_nonface':sum(r[4] for r in rr if not r[2])/nn,'stage_exits':{str(k):{'faces':sum(r[5]==k and r[2]==1 for r in rr),'nonfaces':sum(r[5]==k and r[2]==0 for r in rr)} for k in range(1,9)}}
 groups=[]
 for sc,g in itertools.groupby(sorted(rr,key=lambda x:-x[3]),lambda x:x[3]):
  g=list(g);groups.append((sc,sum(x[2] for x in g),sum(1-x[2] for x in g)))
 tp=fp=0;area=0;points=[(0,0,math.inf)]
 for sc,p,n in groups:
  area+=(tp+p/2)*n;tp+=p;fp+=n;points.append((tp,fp,sc))
 out['auc']=area/(np*nn)
 for rate in [.01,.05,.1]:
  budget=math.floor(nn*rate);tp,fp,sc=max((v for v in points if v[1]<=budget),key=lambda x:(x[0],-x[1]));out[str(rate)]={'detection':tp/np,'precision':tp/(tp+fp) if tp+fp else 0,'tp':tp,'fp':fp,'fp_budget':budget,'threshold':sc if math.isfinite(sc) else None}
 return out
best=None
for ci,(name,target) in enumerate([('A',.5),('B',.35),('C',.2)]):
 p,secs=run(['./train','--build','../train/images','../train/labels.txt',f'candidate_{name}.model',str(target)])
 open(f'construction_{name}.log','w').write(p.stderr)
 st=layout(f'candidate_{name}.model');rec={'candidate':name,'target':target,'stages':st,'wall_seconds':secs,'details':p.stderr};log['constructions'].append(rec)
 json.dump(log,open('search_log.json','w'),indent=2)
 print(name,st,secs,flush=True)
 for gi,(r,o) in enumerate(itertools.product([.995,.998,.999],[0,.25,.5])):
  p,secs=run(['./train','--sweep','../train/images','../train/labels.txt',f'candidate_{name}.model',str(r),str(o)])
  rows=[]
  for line in p.stdout.splitlines():
   a=line.split();rows.append((a[0],int(a[1]),int(a[2]),float(a[3]),int(a[4]),int(a[5])))
  sm=metrics(rows,2);cm=metrics(rows,1);d=sm['0.01']['detection'];cost=sm['mean_weak_nonface'];total=sum(x['weak'] for x in st)
  key=(0,cost,-d,total,ci,gi) if d>=.99 else (1,-d,cost,total,ci,gi)
  item={'candidate':name,'retention':r,'offset':o,'selection':sm,'calibration':cm,'total_weak':total};log['sweeps'].append(item)
  if best is None or key<best[0]:best=(key,item,[x['weak'] for x in st])
  json.dump(log,open('search_log.json','w'),indent=2)
 log['construction_count']=len(log['constructions'])
key,item,sizes=best
config={'candidate':item['candidate'],'stage_sizes':sizes,'retention':item['retention'],'offset':item['offset'],'construction_count':3,'frozen':True}
json.dump(config,open('selected_config.json','w'),indent=2)
log['final_choice']=config;json.dump(log,open('search_log.json','w'),indent=2)
print('FROZEN',config,flush=True)
