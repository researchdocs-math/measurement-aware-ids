"""Controlled replay experiments for configuration-conditioned shift monitoring."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
import argparse,hashlib,itertools,json,pathlib,time
import numpy as np,pandas as pd,torch
from export import ROOT,config_name
from evaluate import load_features,transform,seedof
BASE='p1_i60_a600'

def signatures(folders,config,seed=11):
 parts=[]
 for folder in folders:
  n=len(np.load(folder/'units.npz')['y']);x,u=load_features(folder,config,seed);count=np.bincount(u,minlength=n);sig=np.zeros((n,17),np.float32)
  np.add.at(sig[:,:16],u,transform(x));sig[:,:16]/=np.maximum(count[:,None],1);sig[:,16]=np.log1p(count);parts.append((sig,count>0))
 return np.concatenate([x[0] for x in parts]),np.concatenate([x[1] for x in parts])

def weights(y,cap,mask,prior):
 w=np.zeros(len(y))
 for cls,mass in [(0,1-prior),(1,prior)]:
  active=[c for c in np.unique(cap) if np.any(mask&(cap==c)&(y==cls))]
  for c in active:
   ix=mask&(cap==c)&(y==cls);w[ix]=mass/len(active)/ix.sum()
 return w/w.sum()

def batch_means(phi,observed,indices,device):
 values=torch.as_tensor(phi,device=device);obs=torch.as_tensor(observed.astype(np.float32),device=device);rows=[]
 with torch.no_grad():
  for start in range(0,len(indices),32):
   ids=torch.as_tensor(indices[start:start+32],dtype=torch.long,device=device);mask=obs[ids]
   means=(values[ids]*mask[:,:,None]).sum(1)/mask.sum(1).clamp(min=1)[:,None]
   rows.append(means.cpu().numpy())
 return np.concatenate(rows)

def main(device):
 tic=time.perf_counter();spec=json.loads((ROOT/'configs/experiment.json').read_text());grid=spec['measurement_grid'];configs=list(itertools.product(grid['retention_probability'],grid['idle_timeout_seconds'],grid['active_timeout_seconds']));results=[];metadata=[]
 for corpus in ['IoT-23','Kitsune']:
  folders=sorted(p for p in (ROOT/'data/features').iterdir() if p.is_dir() and p.name.startswith('Kitsune')==(corpus=='Kitsune'))
  metas=[dict(np.load(f/'units.npz')) for f in folders];y=np.concatenate([m['y'] for m in metas]);cap=np.concatenate([np.full(len(m['y']),i) for i,m in enumerate(metas)])
  rng=np.random.default_rng(seedof(corpus));train=np.zeros(len(y),bool)
  # Split within capture/class so the experiment tests a common population
  # mixture on disjoint windows. This is a controlled resampling experiment.
  for c in np.unique(cap):
   for cls in [0,1]:
    ids=np.flatnonzero((cap==c)&(y==cls));rng.shuffle(ids);train[ids[:max(1,len(ids)//2)]]=True
  test=~train;w0=weights(y,cap,train,.1);wt0=weights(y,cap,test,.1);wt1=weights(y,cap,test,.5)
  calidx=rng.choice(len(y),(1000,512),p=w0);nullidx=rng.choice(len(y),(300,512),p=wt0);shiftidx=rng.choice(len(y),(300,512),p=wt1)
  base_sig,base_obs=signatures(folders,BASE)
  mu=base_sig[train].mean(0);sd=base_sig[train].std(0);sd=np.maximum(sd,.1)
  W=rng.normal(size=(17,128)).astype(np.float32)/np.sqrt(17);b=rng.uniform(0,2*np.pi,128).astype(np.float32)
  def embed(sig):
   z=np.clip((sig-mu)/sd,-5,5).astype(np.float32)
   with torch.no_grad():out=(np.sqrt(2/128)*torch.cos(torch.as_tensor(z,device=device)@torch.as_tensor(W,device=device)+torch.as_tensor(b,device=device))).cpu().numpy()
   return out
  phibase=embed(base_sig);qbase=(phibase*w0[:,None]).sum(0)
  bc=batch_means(phibase,base_obs,calidx,device);cutbase=np.quantile(((bc-qbase)**2).sum(1),.95,method='higher')
  # Record an independent CPU comparison for the Metal embedding.
  cpu=np.sqrt(2/128)*np.cos(np.clip((base_sig[:1000]-mu)/sd,-5,5)@W+b)
  metadata.append({'corpus':corpus,'embedding_max_abs_cpu_difference':float(np.max(np.abs(cpu-phibase[:1000]))),'reference_windows':int(train.sum()),'test_windows':int(test.sum())})
  for p,idle,active in configs:
   config=config_name(p,idle,active);sig,obs=signatures(folders,config);phi=embed(sig)
   q=(phi*(w0*obs)[:,None]).sum(0)/(w0*obs).sum()
   cal=batch_means(phi,obs,calidx,device)
   cutmatched=np.quantile(((cal-q)**2).sum(1),.95,method='higher');cutreset=np.quantile(((cal-qbase)**2).sum(1),.95,method='higher')
   for behavior,indices in [('unchanged',nullidx),('changed',shiftidx)]:
    # Use independently sampled packet realizations for current traffic.
    for seed in ([11] if p==1 else [23,47,83,131]):
     cs,co=signatures(folders,config,seed);cp=embed(cs);means=batch_means(cp,co,indices,device);dm=((means-q)**2).sum(1);df=((means-qbase)**2).sum(1)
     decisions={'Fixed reference':df>cutbase,'Threshold reset':df>cutreset,'Matched reference':dm>cutmatched,'Configuration blanking':(df>cutbase) if config==BASE else np.zeros(len(df),bool)}
     for method,alarm in decisions.items():
      for rep,a in enumerate(alarm):results.append({'corpus':corpus,'config':config,'p':p,'idle':idle,'active':active,'behavior':behavior,'seed':seed,'replicate':rep,'method':method,'alarm':int(a),'observed_units':int(co[indices[rep]].sum())})
   print(corpus,config,'finished',flush=True)
  pd.DataFrame(results).to_csv(ROOT/'results/drift_trials.csv',index=False)
 (ROOT/'results/drift_runtime.json').write_text(json.dumps({'device':device,'seconds':time.perf_counter()-tic,'replicates':300,'calibration_replicates':1000,'latent_units_per_batch':512,'rff_dimensions':128,'checks':metadata},indent=2))
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--device',default='mps' if torch.backends.mps.is_available() else 'cpu');args=parser.parse_args();main(args.device)
