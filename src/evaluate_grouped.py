"""Complementary evaluation with disjoint five-tuple groups in each capture."""
from evaluate import *

def run_grouped(task):
 corpus,splitseed=task;out=ROOT/'results'/f'grouped_{corpus}_{splitseed}.csv'
 if out.exists():return str(out)
 folders=sorted(p for p in (ROOT/'data/features').iterdir() if p.is_dir() and p.name.startswith('Kitsune')==(corpus=='Kitsune'))
 metadata={p.name:dict(np.load(p/'units.npz')) for p in folders};parts={};testmasks={};partition_counts=[]
 for folder in folders:
  m=metadata[folder.name];rng=np.random.default_rng(seedof(folder.name)+splitseed);groups=np.unique(m['tuple_group']);rng.shuffle(groups);a=int(.6*len(groups));b=int(.8*len(groups))
  tr=np.isin(m['tuple_group'],groups[:a]);cal=np.isin(m['tuple_group'],groups[a:b]);testmasks[folder.name]=np.isin(m['tuple_group'],groups[b:]);choose=[]
  for cls in [0,1]:
   ids=np.flatnonzero(tr&(m['y']==cls));choose.extend(rng.choice(ids,min(3000,len(ids)),replace=False))
  tr[:]=False;tr[choose]=True;parts[folder.name]=(tr,cal)
  partition_counts.append({'capture':folder.name,'train':int(tr.sum()),'calibration':int(cal.sum()),'test':int(testmasks[folder.name].sum())})
 base=fit(train_data(folders,metadata,parts,BASE));basecal=calibration(base,folders,metadata,parts,BASE)
 spec=json.loads((ROOT/'configs/experiment.json').read_text());grid=spec['measurement_grid'];configs=list(itertools.product(grid['retention_probability'],grid['idle_timeout_seconds'],grid['active_timeout_seconds']))
 aug=[train_data(folders,metadata,parts,config_name(p,i,a),extra=np.array([np.log(p),np.log(i),np.log(a)])) for p,i,a in configs]
 pooled=fit(tuple(np.concatenate([x[j] for x in aug]) for j in range(3)));del aug;rows=[]
 for p,idle,active in configs:
  config=config_name(p,idle,active);replay=fit(train_data(folders,metadata,parts,config,True));unweighted=fit(train_data(folders,metadata,parts,config,False));extra=np.array([np.log(p),np.log(idle),np.log(active)])
  cals={'Frozen':basecal,'Count correction':basecal,'Threshold replay':calibration(base,folders,metadata,parts,config),'Pooled + configuration':calibration(pooled,folders,metadata,parts,config,extra=extra),'Replay unweighted':calibration(unweighted,folders,metadata,parts,config),'Replay weighted':calibration(replay,folders,metadata,parts,config)}
  thresholds={method:{alpha:threshold(sc,labels,alpha) for alpha in [.01,.05]} for method,(sc,labels) in cals.items()}
  for seed in ([11] if p==1 else spec['sampling_seeds']):
   for target in folders:
    meta=metadata[target.name];y=meta['y'];n=len(y);test=testmasks[target.name];x,u=load_features(target,config,seed);observed=np.bincount(u,minlength=n)>0;sx=x.copy();sx[:,:2]/=p
    scores={'Frozen':maxscore(base,x,u,n),'Count correction':maxscore(base,sx,u,n),'Threshold replay':maxscore(base,x,u,n),'Pooled + configuration':maxscore(pooled,x,u,n,extra),'Replay unweighted':maxscore(unweighted,x,u,n),'Replay weighted':maxscore(replay,x,u,n)}
    for method,score in scores.items():
     for alpha in [.01,.05]:rows.append({'corpus':corpus,'capture':target.name,'splitseed':splitseed,'config':config,'p':p,'idle':idle,'active':active,'seed':seed,'method':method,'alpha':alpha,**metrics(score[test],y[test],thresholds[method][alpha],observed[test])})
  print(corpus,splitseed,config,flush=True)
 pd.DataFrame(rows).to_csv(out,index=False);out.with_suffix('.json').write_text(json.dumps(partition_counts,indent=2));return str(out)
if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=4);args=parser.parse_args()
 with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
  for p in pool.map(run_grouped,[(c,s) for c in ['IoT-23','Kitsune'] for s in [7919,104729,130363]]):print(p,flush=True)
