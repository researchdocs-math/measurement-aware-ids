"""Matched-population intervention control with independent batch draws."""
from drift import *
def run(device):
 rows=[];tic=time.perf_counter();spec=json.loads((ROOT/'configs/experiment.json').read_text());grid=spec['measurement_grid']
 for corpus in ['IoT-23','Kitsune']:
  folders=sorted(p for p in (ROOT/'data/features').iterdir() if p.is_dir() and p.name.startswith('Kitsune')==(corpus=='Kitsune'))
  metas=[dict(np.load(p/'units.npz')) for p in folders];y=np.concatenate([m['y'] for m in metas]);cap=np.concatenate([np.full(len(m['y']),i) for i,m in enumerate(metas)])
  rng=np.random.default_rng(seedof(corpus)+2026);w0=weights(y,cap,np.ones(len(y),bool),.1);w1=weights(y,cap,np.ones(len(y),bool),.5)
  calidx=rng.choice(len(y),(1000,512),p=w0);idx0=rng.choice(len(y),(300,512),p=w0);idx1=rng.choice(len(y),(300,512),p=w1)
  sb,ob=signatures(folders,BASE);mu=sb.mean(0);sd=np.maximum(sb.std(0),.1);W=rng.normal(size=(17,128)).astype(np.float32)/np.sqrt(17);b=rng.uniform(0,2*np.pi,128).astype(np.float32)
  def embed(sig):
   with torch.no_grad():return (np.sqrt(2/128)*torch.cos(torch.as_tensor(np.clip((sig-mu)/sd,-5,5),device=device)@torch.as_tensor(W,device=device)+torch.as_tensor(b,device=device))).cpu().numpy()
  pb=embed(sb);qb=(pb*w0[:,None]).sum(0);calbase=batch_means(pb,ob,calidx,device);cutbase=np.quantile(((calbase-qb)**2).sum(1),.95,method='higher')
  for p,idle,active in itertools.product(grid['retention_probability'],grid['idle_timeout_seconds'],grid['active_timeout_seconds']):
   config=config_name(p,idle,active)
   for seed in ([11] if p==1 else [11,23,47,83,131]):
    sig,obs=signatures(folders,config,seed);phi=embed(sig);q=(phi*(w0*obs)[:,None]).sum(0)/(w0*obs).sum();cal=batch_means(phi,obs,calidx,device)
    cut=np.quantile(((cal-q)**2).sum(1),.95,method='higher');cutreset=np.quantile(((cal-qb)**2).sum(1),.95,method='higher')
    for behavior,idx in [('unchanged',idx0),('changed',idx1)]:
     mean=batch_means(phi,obs,idx,device);dm=((mean-q)**2).sum(1);df=((mean-qb)**2).sum(1)
     alarms={'Fixed reference':df>cutbase,'Threshold reset':df>cutreset,'Matched reference':dm>cut,'Configuration blanking':df>cutbase if config==BASE else np.zeros(len(df),bool)}
     for method,aa in alarms.items():
      for rep,a in enumerate(aa):rows.append({'corpus':corpus,'config':config,'p':p,'idle':idle,'active':active,'behavior':behavior,'seed':seed,'replicate':rep,'method':method,'alarm':int(a)})
   print(corpus,config,flush=True)
 pd.DataFrame(rows).to_csv(ROOT/'results/drift_control.csv',index=False)
 (ROOT/'results/drift_control_runtime.json').write_text(json.dumps({'device':device,'seconds':time.perf_counter()-tic,'population':'Common empirical population; separate independently drawn calibration, unchanged-traffic, and changed-traffic batches. Each sampling realization is calibrated independently.'},indent=2))
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--device',default='mps' if torch.backends.mps.is_available() else 'cpu');args=parser.parse_args();run(args.device)
