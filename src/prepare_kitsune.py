"""Decode Kitsune's packet labels, asserting one label per original packet."""
import collections,concurrent.futures,json,pathlib,time
import dpkt,numpy as np
from prepare import canonical,ROOT

def run(folder):
 tic=time.perf_counter();out=ROOT/'data/packets'/(folder.name+'.npz')
 if out.exists():return json.loads(out.with_suffix('.json').read_text())
 labelpath=next(folder.glob('*labels.csv'))
 if folder.name=='Kitsune-Mirai-Botnet':labels=np.loadtxt(labelpath,dtype=np.int8)
 else:labels=np.loadtxt(labelpath,delimiter=',',skiprows=1,usecols=1,dtype=np.int8)
 pcap=next(p for p in folder.glob('*pcap*') if p.suffix in ('.pcap','.pcapng'));rows=[];keys={};totals=collections.Counter();t0=None
 with pcap.open('rb') as f:
  magic=f.read(4);f.seek(0);reader=dpkt.pcapng.Reader(f) if magic==b'\x0a\x0d\x0d\x0a' else dpkt.pcap.Reader(f)
  for index,(ts,buf) in enumerate(reader):
   totals['raw_packets']+=1;ts=float(ts)
   if t0 is None:t0=ts
   if index>=len(labels):raise ValueError('More packets than labels: '+folder.name)
   try:
    ip=dpkt.ethernet.Ethernet(buf).data
    if not isinstance(ip,dpkt.ip.IP) or ip.p not in (6,17):totals['non_ipv4_tcp_udp']+=1;continue
    if ip.mf or ip.offset:totals['fragmented_ipv4']+=1;continue
    l4=ip.data
    if not isinstance(l4,(dpkt.tcp.TCP,dpkt.udp.UDP)):totals['truncated_transport']+=1;continue
    k=canonical(ip.src,l4.sport,ip.dst,l4.dport,ip.p)
    if k not in keys:keys[k]=len(keys)
    rows.append((ts-t0,keys[k],int(ip.len),int((ip.src,l4.sport)!=k[0]),int(l4.flags) if ip.p==6 else 0,ip.p,labels[index],index))
   except (dpkt.UnpackError,ValueError,IndexError,AttributeError):totals['parse_errors']+=1
 assert totals['raw_packets']==len(labels),(folder.name,totals['raw_packets'],len(labels))
 a=np.asarray(rows,dtype=np.float64);del rows
 _,uid=np.unique(np.stack([a[:,1].astype(np.int64),np.floor(a[:,0]/600).astype(np.int64)],axis=1),axis=0,return_inverse=True)
 order=np.lexsort((a[:,7],a[:,0],uid));a=a[order];uid=uid[order]
 begins=np.r_[0,np.flatnonzero(np.diff(uid))+1];ends=np.r_[begins[1:],len(a)]
 low=np.minimum.reduceat(a[:,6],begins).astype(np.int8);high=np.maximum.reduceat(a[:,6],begins).astype(np.int8);good=np.ones(len(low),dtype=bool)
 totals['all_communication_windows']=len(begins);totals['mixed_labeled_windows']=int((low!=high).sum())
 select=np.repeat(good,ends-begins);unit=np.repeat(np.cumsum(good)-1,ends-begins)[select].astype(np.int32);a=a[select]
 begin=np.r_[0,np.flatnonzero(np.diff(unit))+1];end=np.r_[begin[1:],len(a)]
 np.savez_compressed(out,t=a[:,0],length=a[:,2].astype(np.float32),direction=a[:,3].astype(np.int8),flags=a[:,4].astype(np.uint8),proto=a[:,5].astype(np.uint8),packet_index=a[:,7].astype(np.int64),unit=unit,begin=begin.astype(np.int64),end=end.astype(np.int64),y=high[good],block=np.floor(a[begin,0]/600).astype(np.int32),tuple_group=a[begin,1].astype(np.int64),unit_start=a[begin,0])
 summary={'capture':folder.name,**dict(totals),'eligible_windows':len(begin),'benign_windows':int((high[good]==0).sum()),'attack_windows':int((high[good]==1).sum()),'retained_packets':len(a),'duration_seconds':float(a[:,0].max()),'seconds':time.perf_counter()-tic}
 out.with_suffix('.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True);return summary
if __name__=='__main__':
 folders=sorted(p for p in (ROOT/'data/raw').glob('Kitsune-*') if list(p.glob('*labels.csv')) and any(p.glob('*.pcap*')))
 with concurrent.futures.ProcessPoolExecutor(max_workers=3) as pool:results=list(pool.map(run,folders))
 (ROOT/'results/kitsune_audit.json').write_text(json.dumps(results,indent=2))
