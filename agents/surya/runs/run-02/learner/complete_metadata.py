import os,subprocess,itertools,json
for name in 'ABC':
 for r,o in itertools.product([.995,.998,.999],[0,.25,.5]):
  path=f'candidate_{name}.model.r{r}.o{o}.json'
  if os.path.exists(path):continue
  if not os.path.exists(f'candidate_{name}.model'):continue
  with open('metadata_last_rows.txt','w') as f:
   subprocess.run(['./train','--sweep','../train/images','../train/labels.txt',f'candidate_{name}.model',str(r),str(o)],stdout=f,stderr=subprocess.DEVNULL,check=True)
print('Metadata completed; no stumps were refitted.')
