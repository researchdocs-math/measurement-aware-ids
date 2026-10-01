from pathlib import Path
import json,glob,numpy as np,pandas as pd
R=Path(__file__).resolve().parents[1];O=R/'results/summary';O.mkdir(exist_ok=True)
methods=['Frozen','Count correction','Threshold replay','Pooled + configuration','Replay unweighted','Replay weighted']
allframes=[]
for regime,pattern in [('Held-out capture','classification_*.csv'),('Grouped holdout','grouped_*.csv')]:
 d=pd.concat([pd.read_csv(f) for f in (R/'results').glob(pattern)],ignore_index=True);d['regime']=regime;d['visibility']=d.observed_positive/d.positive_units
 # Each capture/configuration has equal weight; seeds are Monte Carlo repeats.
 cols=['regime','corpus','capture','config','p','idle','active','method','alpha']
 values=['recall','fpr','tp','fp','positive_units','negative_units','observed_positive','observed_negative','observed_recall','visibility']
 a=d.groupby(cols,dropna=False)[values].mean().reset_index()
 allframes.append(a)
a=pd.concat(allframes,ignore_index=True);a.to_csv(O/'classification_capture_config.csv',index=False)
b=a.groupby(['regime','corpus','p','method','alpha'])[['recall','fpr','visibility','observed_recall']].mean().reset_index();b.to_csv(O/'classification_by_p.csv',index=False)
c=a.groupby(['regime','corpus','capture','method','alpha'])[['recall','fpr']].mean().reset_index();c.to_csv(O/'classification_capture_mean.csv',index=False)
agg=c.groupby(['regime','corpus','method','alpha'])[['recall','fpr']].mean().reset_index();agg.to_csv(O/'classification_summary.csv',index=False)
# Paired capture bootstrap, not seed-level pseudo-replication.
rng=np.random.default_rng(24680);rows=[]
for (regime,corpus,alpha),d in c.groupby(['regime','corpus','alpha']):
 for left,right in [('Replay weighted','Frozen'),('Pooled + configuration','Replay weighted'),('Replay weighted','Replay unweighted')]:
  for metric in ['recall','fpr']:
   piv=d.pivot(index='capture',columns='method',values=metric);delta=(piv[left]-piv[right]).dropna().to_numpy();boot=rng.choice(delta,(10000,len(delta)),replace=True).mean(1)
   rows.append(dict(regime=regime,corpus=corpus,alpha=alpha,left=left,right=right,metric=metric,n=len(delta),difference=delta.mean(),low=np.quantile(boot,.025),high=np.quantile(boot,.975)))
pd.DataFrame(rows).to_csv(O/'paired_capture_intervals.csv',index=False)
drift=[]
for name,regime in [('drift_control','Common population'),('drift_trials','Reference mismatch')]:
 d=pd.read_csv(R/'results'/f'{name}.csv')
 d=d.groupby(['corpus','config','p','idle','active','behavior','method']).alarm.mean().reset_index();d['regime']=regime;drift.append(d)
d=pd.concat(drift,ignore_index=True);d.to_csv(O/'drift_configuration.csv',index=False)
s=d[d.config!='p1_i60_a600'].groupby(['regime','corpus','behavior','method']).alarm.mean().reset_index();s.to_csv(O/'drift_summary.csv',index=False)
vis=[];fragment=[];audit=[]
all_audit=[json.load(open(p)) for p in sorted((R/'data/packets').glob('*.json'))]
for f in sorted((R/'data/features').iterdir()):
 if not f.is_dir():continue
 m=dict(np.load(f/'units.npz'));corpus='Kitsune' if f.name.startswith('Kitsune') else 'IoT-23';y=m['y'];n=m['packet_count']
 row=next(x.copy() for x in all_audit if x['capture']==f.name);row.update(selected_benign=int((y==0).sum()),selected_attack=int((y==1).sum()),corpus=corpus);audit.append(row)
 for p in [1,.5,.1,.01]:
  seeds=[11] if p==1 else [11,23,47,83,131]
  for label in [0,1]:
   if not (y==label).any():continue
   expected=np.mean(-np.expm1(n[y==label]*np.log1p(-p))) if p<1 else 1.
   vals=[]
   for seed in seeds:
    ex=np.load(f/f'p{p:g}_i60_a600_s{seed}.npz');obs=np.bincount(ex['unit'],minlength=len(y))>0;vals.append(obs[y==label].mean())
   vis.append(dict(corpus=corpus,capture=f.name,p=p,label=label,expected=expected,observed=np.mean(vals),min=np.min(vals),max=np.max(vals)))
  for idle in [5,30,60]:
   for active in [30,120,600]:
    nums=[]
    for seed in seeds:
     ex=np.load(f/f'p{p:g}_i{idle}_a{active}_s{seed}.npz');counts=np.bincount(ex['unit'],minlength=len(y));nums.append((counts.mean(),np.mean(counts[counts>0]),counts.max(),np.mean(counts>1)))
    vals=np.mean(nums,axis=0);fragment.append(dict(corpus=corpus,capture=f.name,p=p,idle=idle,active=active,records_per_unit=vals[0],records_per_observed=vals[1],max_fragments=vals[2],multiple_fraction=vals[3]))
pd.DataFrame(vis).to_csv(O/'visibility.csv',index=False);pd.DataFrame(fragment).to_csv(O/'fragmentation.csv',index=False);pd.DataFrame(audit).to_csv(O/'capture_audit.csv',index=False)
# Consistency checks and machine-readable headline values.
raw=pd.concat([pd.read_csv(f) for f in (R/'results').glob('classification_*.csv')]);p=raw[(raw.p==1)&(raw.alpha==.01)].pivot(index=['capture','config','seed'],columns='method',values=['tp','fp'])
assert np.array_equal(p['tp']['Frozen'],p['tp']['Count correction']);assert np.array_equal(p['fp']['Frozen'],p['fp']['Count correction'])
base=a[(a.config=='p1_i60_a600')].pivot(index=['regime','corpus','capture','alpha'],columns='method',values=['tp','fp'])
assert np.array_equal(base['tp']['Frozen'],base['tp']['Replay weighted']);assert np.array_equal(base['fp']['Frozen'],base['fp']['Replay weighted'])
assert (a.tp<=a.observed_positive+1e-9).all()
print('All analysis invariants passed.');print(agg[agg.alpha==.01].to_string(index=False));print(s.to_string(index=False))
