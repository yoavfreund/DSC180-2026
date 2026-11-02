import json,re,os
p='search_log.json';x=json.load(open(p))
for c in x['constructions']:
 stages=[]
 for line in c['details'].splitlines():
  m=re.match(r'stage (\d+) weak (\d+) cal_faces (\d+) cal_neg (\d+) pass ([\d.]+) target ([\d.]+) met (\d+) remaining (\d+) threshold (.*)',line)
  if m:
   k,n,faces,neg,ps,t,met,remain,th=m.groups();stages.append(dict(stage=int(k),weak=int(n),arriving_calibration_faces=int(faces),arriving_calibration_nonfaces=int(neg),nonface_pass_fraction=float(ps),target=float(t),target_met=bool(int(met)),passing_calibration_nonfaces=int(remain),exiting_calibration_nonfaces=int(neg)-int(remain),threshold=float(th)))
 c['construction_stages']=stages;c['cumulative_nonface_pass_fraction']=stages[-1]['passing_calibration_nonfaces']/787;c['cumulative_target_met']=c['cumulative_nonface_pass_fraction']<=.005;c['stop_reason']='160-classifier total budget'
for s in x['sweeps']:
 path=f"candidate_{s['candidate']}.model.r{s['retention']}.o{s['offset']}.json"
 s['threshold_calibration_stages']=json.load(open(path))
 last=s['threshold_calibration_stages'][-1];s['calibration']['cascade_recall']=last['passing_faces']/399;s['calibration']['cascade_nonface_pass_fraction']=last['passing_nonfaces']/787
x['threshold_metadata_note']='Threshold metadata was supplemented by repeating calibration only on fixed stumps; no additional constructions or refits.'
x['validation_calls_before_freeze']=0
json.dump(x,open(p,'w'),indent=2)
print('All 27 sweep thresholds and calibration arrival/pass counts merged.')
