import random,json
rows=[l.split() for l in open('../train/labels.txt') if l.strip()]
r=random.Random(7321); r.shuffle(rows)
sel=[x for i,x in enumerate(rows) if (i+1)%5==0]; rem=[x for i,x in enumerate(rows) if (i+1)%5!=0]
random.Random(2026).shuffle(rem)
cal=[x for i,x in enumerate(rem) if (i+1)%4==0]; fit=[x for i,x in enumerate(rem) if (i+1)%4!=0]
for name,rr in [('fitting',fit),('calibration',cal),('selection',sel)]:
 with open(name+'.txt','w') as f:
  for x in rr:f.write(' '.join(x)+'\n')
plan=dict(candidates={'A':.5,'B':.35,'C':.2},construction_retention=.998,stage_min=2,stage_max=64,stages_max=8,total_max=160,cumulative_negative_target=.005,retentions=[.995,.998,.999],offsets=[0,.25,.5],split={'selection_seed':7321,'selection_rule':'one-based every fifth','calibration_seed':2026,'calibration_rule':'one-based every fourth remaining'},selection_rule='tied-score ROC, floor(0.01*Nneg) FP budget; eligible detection >= .99 then minimum negative weak evaluations, higher detection, fewer weak; otherwise higher detection, lower negative evaluations, fewer weak; A/B/C then grid order',counts={n:{'originals':len(rr),'faces':sum(int(x[1]) for x in rr)} for n,rr in [('fitting',fit),('calibration',cal),('selection',sel)]},empty_arrivals='No calibration faces: threshold -1e30 (retain all); zero calibration negatives: pass target satisfied. Zero fitting negatives: append deterministic zero-alpha stumps if required for frozen/minimum architecture; no weight fitting possible.')
json.dump(plan,open('experiment_plan.json','w'),indent=2)
print(plan['counts'])
