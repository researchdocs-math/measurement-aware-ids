"""Capture-held-out evaluation: no target labels enter fitting or calibration."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import concurrent.futures,hashlib,itertools,json,pathlib,resource,time
import numpy as np,pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from export import ROOT,config_name
BASE='p1_i60_a600'

def seedof(s):return int.from_bytes(hashlib.sha256(s.encode()).digest()[:4],'little')
def load_features(folder,config,seed=11):
 if config.startswith('p1_'):seed=11
 a=np.load(folder/f'{config}_s{seed}.npz');return a['X'].copy(),a['unit'].copy()
def transform(x):
 z=x.copy();z[:,:10]=np.log1p(z[:,:10]);return z

def maxscore(model,x,u,n,extra=None):
 scores=np.full(n,-1.,dtype=float)
 if len(x):
  z=transform(x)
  if extra is not None:z=np.column_stack([z,np.broadcast_to(extra,(len(z),len(extra)))])
  np.maximum.at(scores,u,model.predict_proba(z)[:,1])
 return scores

def threshold(scores,y,alpha):
 s=np.sort(scores[y==0]);n=len(s)
 if not n:return 1.
 # Strict exceedance handles score ties conservatively.
 k=int(np.ceil((n+1)*(1-alpha)))
 return float(s[k-1]) if k<=n else 1.

def metrics(scores,y,threshold_value,observed):
 pred=scores>threshold_value;pos=y==1;neg=~pos
 tp=int((pred&pos).sum());fp=int((pred&neg).sum());np_=int(pos.sum());nn=int(neg.sum());obspos=int((pos&observed).sum())
 return {'tp':tp,'fp':fp,'positive_units':np_,'negative_units':nn,'observed_positive':obspos,'observed_negative':int((neg&observed).sum()),'recall':tp/np_ if np_ else np.nan,'fpr':fp/nn if nn else np.nan,'observed_recall':tp/obspos if obspos else np.nan,'precision':tp/(tp+fp) if tp+fp else np.nan,'threshold':threshold_value}

def make_partitions(folder,meta):
 rng=np.random.default_rng(seedof(folder.name));n=len(meta['y'])
 groups=meta.get('tuple_group',meta['source_unit'])
 g=np.unique(groups);calgroups=rng.choice(g,max(1,int(np.ceil(.2*len(g)))),replace=False)
 cal=np.isin(groups,calgroups);train=~cal
 # Identical labeled-unit budget for all methods.
 choose=[]
 for y in [0,1]:
  ids=np.flatnonzero(train&(meta['y']==y));choose.extend(rng.choice(ids,min(len(ids),3000),replace=False))
 train=np.zeros(n,bool);train[choose]=True
 return train,cal

def train_data(folders,metadata,partitions,config,weighted=True,extra=None):
 XX=[];YY=[];WW=[]
 for folder in folders:
  x,u=load_features(folder,config);meta=metadata[folder.name];train,_=partitions[folder.name];take=train[u];x=x[take];u=u[take]
  if not len(u):continue
  y=meta['y'][u];count=np.bincount(u,minlength=len(meta['y']))
  w=1/count[u] if weighted else np.ones(len(u))
  # Equal total weight for each available class within each source capture.
  for c in [0,1]:
   ix=y==c
   if ix.any():w[ix]/=w[ix].sum()
  z=transform(x)
  if extra is not None:z=np.column_stack([z,np.broadcast_to(extra,(len(z),len(extra)))])
  XX.append(z);YY.append(y);WW.append(w)
 return np.concatenate(XX),np.concatenate(YY),np.concatenate(WW)

def fit(data):
 x,y,w=data;model=ExtraTreesClassifier(n_estimators=64,max_depth=16,min_samples_leaf=5,max_features=1.,n_jobs=2,random_state=20260930)
 model.fit(x,y,sample_weight=w*len(w)/w.sum())
 # Serial prediction fixes tree-summation order at strict threshold ties.
 model.set_params(n_jobs=1)
 return model

def calibration(model,folders,metadata,partitions,config,correction=1.,extra=None):
 scores=[];yy=[]
 for folder in folders:
  x,u=load_features(folder,config);x[:,0:2]/=correction;meta=metadata[folder.name];_,cal=partitions[folder.name]
  scores.append(maxscore(model,x,u,len(meta['y']),extra)[cal]);yy.append(meta['y'][cal])
 return np.concatenate(scores),np.concatenate(yy)

def run_fold(target_name):
 tic=time.perf_counter();root=ROOT/'data/features';target=root/target_name
 corpus='Kitsune' if target_name.startswith('Kitsune') else 'IoT-23'
 folders=sorted(p for p in root.iterdir() if p.is_dir() and p.name!=target_name and (p.name.startswith('Kitsune')==(corpus=='Kitsune')))
 metadata={p.name:dict(np.load(p/'units.npz')) for p in folders+[target]};parts={p.name:make_partitions(p,metadata[p.name]) for p in folders}
 resultfile=ROOT/'results'/f'classification_{target_name}.csv'
 if resultfile.exists():return str(resultfile)
 base=fit(train_data(folders,metadata,parts,BASE));basecal=calibration(base,folders,metadata,parts,BASE)
 spec=json.loads((ROOT/'configs/experiment.json').read_text());grid=spec['measurement_grid'];configs=list(itertools.product(grid['retention_probability'],grid['idle_timeout_seconds'],grid['active_timeout_seconds']))
 # Pooled, configuration-conditioned training uses all configurations and the
 # same labeled source units. Each configuration has equal total fit weight.
 aug=[]
 for p,idle,active in configs:
  data=train_data(folders,metadata,parts,config_name(p,idle,active),extra=np.array([np.log(p),np.log(idle),np.log(active)]))
  aug.append(data)
 pooldata=tuple(np.concatenate([x[j] for x in aug]) for j in range(3));pooled=fit(pooldata);del pooldata,aug
 y=metadata[target_name]['y'];n=len(y);rows=[];timings=[]
 for p,idle,active in configs:
  config=config_name(p,idle,active);start=time.perf_counter()
  replay=fit(train_data(folders,metadata,parts,config,True));unweighted=fit(train_data(folders,metadata,parts,config,False))
  extra=np.array([np.log(p),np.log(idle),np.log(active)])
  cals={'Frozen':basecal,'Count correction':calibration(base,folders,metadata,parts,BASE),'Threshold replay':calibration(base,folders,metadata,parts,config),'Pooled + configuration':calibration(pooled,folders,metadata,parts,config,extra=extra),'Replay unweighted':calibration(unweighted,folders,metadata,parts,config),'Replay weighted':calibration(replay,folders,metadata,parts,config)}
  thresholds={method:{alpha:threshold(sc,labels,alpha) for alpha in [.01,.05]} for method,(sc,labels) in cals.items()}
  for seed in ([11] if p==1 else spec['sampling_seeds']):
   x,u=load_features(target,config,seed);observed=np.bincount(u,minlength=n)>0
   sx=x.copy();sx[:,:2]/=p
   scores={'Frozen':maxscore(base,x,u,n),'Count correction':maxscore(base,sx,u,n),'Threshold replay':maxscore(base,x,u,n),'Pooled + configuration':maxscore(pooled,x,u,n,extra),'Replay unweighted':maxscore(unweighted,x,u,n),'Replay weighted':maxscore(replay,x,u,n)}
   for method,score in scores.items():
    for alpha in [.01,.05]:
     rows.append({'corpus':corpus,'capture':target_name,'config':config,'p':p,'idle':idle,'active':active,'seed':seed,'method':method,'alpha':alpha,**metrics(score,y,thresholds[method][alpha],observed)})
  timings.append({'config':config,'seconds':time.perf_counter()-start});print(target_name,config,'done',flush=True)
 pd.DataFrame(rows).to_csv(resultfile,index=False)
 (ROOT/'results'/f'timing_{target_name}.json').write_text(json.dumps({'capture':target_name,'seconds':time.perf_counter()-tic,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'configuration_timings':timings},indent=2))
 return str(resultfile)
if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=4);parser.add_argument('--capture');args=parser.parse_args()
 targets=[args.capture] if args.capture else [p.name for p in sorted((ROOT/'data/features').iterdir()) if p.is_dir()]
 with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
  for out in pool.map(run_fold,targets):print(out,flush=True)
