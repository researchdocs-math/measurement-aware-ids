"""Packet sampling precedes metering; no exported-row sampling is used."""
import concurrent.futures, itertools, json, pathlib, time, hashlib
import numpy as np
from numba import njit
ROOT=pathlib.Path(__file__).resolve().parents[1]
FEATURES=['packets','ip_bytes','duration','mean_length','std_length','min_length','max_length','mean_iat','std_iat','max_iat','reverse_fraction','syn_fraction','ack_fraction','rst_fraction','fin_fraction','is_udp']

@njit(cache=True)
def meter(t,length,direction,flags,proto,begin,end,selected,uniform,p,idle,active):
    # At most one record per retained packet, including at timeout boundaries.
    X=np.empty((len(t),16),np.float32);U=np.empty(len(t),np.int32);Z=np.empty(len(t),np.float64);count=0
    for ui in range(len(selected)):
        u=selected[ui]; n=0;total=0.;sq=0.;mini=0.;maxi=0.;iat=0.;iat2=0.;imax=0.;reverse=0;syn=0;ack=0;rst=0;fin=0;first=0.;last=0.;origin=0;pr=0
        for i in range(begin[u],end[u]+1):
            terminal=i==end[u]
            if not terminal and uniform[i]>=p:continue
            if n>0 and (terminal or t[i]-last>idle or t[i]-first>=active):
                X[count,0]=n;X[count,1]=total;X[count,2]=last-first;X[count,3]=total/n;X[count,4]=np.sqrt(max(0.,sq/n-(total/n)**2));X[count,5]=mini;X[count,6]=maxi
                X[count,7]=iat/max(1,n-1);X[count,8]=np.sqrt(max(0.,iat2/max(1,n-1)-(iat/max(1,n-1))**2));X[count,9]=imax
                X[count,10]=reverse/n;X[count,11]=syn/n;X[count,12]=ack/n;X[count,13]=rst/n;X[count,14]=fin/n;X[count,15]=pr==17
                U[count]=ui;Z[count]=last;count+=1;n=0;total=0.;sq=0.;iat=0.;iat2=0.;imax=0.;reverse=0;syn=0;ack=0;rst=0;fin=0
            if terminal:break
            if n==0:first=t[i];origin=direction[i];mini=length[i];maxi=length[i];pr=proto[i]
            else:
                gap=t[i]-last;iat+=gap;iat2+=gap*gap;imax=max(imax,gap)
            n+=1;total+=length[i];sq+=length[i]*length[i];mini=min(mini,length[i]);maxi=max(maxi,length[i]);reverse+=direction[i]!=origin
            syn+=(flags[i]&2)!=0;ack+=(flags[i]&16)!=0;rst+=(flags[i]&4)!=0;fin+=(flags[i]&1)!=0;last=t[i]
    return X[:count],U[:count],Z[:count]

def config_name(p,idle,active):return f'p{p:g}_i{idle}_a{active}'
def run_capture(args):
    capidx,path=args;capidx=int.from_bytes(hashlib.sha256(path.stem.encode()).digest()[:4],'little');a=np.load(path);seedselect=np.random.default_rng(20260930+capidx)
    # Fixed cap keeps workloads bounded; selection occurs before transformations.
    selected=np.sort(np.concatenate([seedselect.choice(np.flatnonzero(a['y']==c),min(20000,int((a['y']==c).sum())),replace=False) for c in [0,1]]))
    folder=ROOT/'data/features'/path.stem;folder.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(folder/'units.npz',source_unit=selected,tuple_group=a['tuple_group'][selected],y=a['y'][selected],block=a['block'][selected],start=a['unit_start'][selected],packet_count=(a['end']-a['begin'])[selected])
    spec=json.loads((ROOT/'configs/experiment.json').read_text());grid=spec['measurement_grid']; summary=[]
    timing=folder/'export_timing.json'
    if timing.exists(): summary=json.loads(timing.read_text())
    for seed in spec['sampling_seeds']:
        uniform=np.random.default_rng(seed+1009*capidx).random(len(a['t']))
        for p,idle,active in itertools.product(grid['retention_probability'],grid['idle_timeout_seconds'],grid['active_timeout_seconds']):
            # All p=1 exports are identical across sampling seeds.
            if p==1 and seed!=spec['sampling_seeds'][0]:continue
            name=config_name(p,idle,active);out=folder/f'{name}_s{seed}.npz'
            if out.exists():continue
            tic=time.perf_counter();x,u,z=meter(*[a[k] for k in ['t','length','direction','flags','proto','begin','end']],selected,uniform,p,idle,active)
            np.savez_compressed(out,X=x,unit=u,end_time=z)
            summary.append({'capture':path.stem,'config':name,'seed':seed,'records':len(u),'observed_units':len(np.unique(u)),'seconds':time.perf_counter()-tic})
    (folder/'export_timing.json').write_text(json.dumps(summary,indent=2));print(path.stem,len(selected),'units',len(summary),'exports',flush=True)
if __name__=='__main__':
    (ROOT/'configs/features.json').write_text(json.dumps(FEATURES))
    paths=sorted((ROOT/'data/packets').glob('*.npz'))
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=5);args=parser.parse_args()
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:list(pool.map(run_capture,enumerate(paths)))
