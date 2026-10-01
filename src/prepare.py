"""Decode original PCAPs and construct label-audited communication windows."""
import bisect, collections, concurrent.futures, gzip, json, pathlib, socket, time
import dpkt
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1]

def canonical(src,sport,dst,dport,proto):
    a=(src,int(sport)); b=(dst,int(dport))
    return (a,b,int(proto)) if a<=b else (b,a,int(proto))

def read_labels(path):
    labels=collections.defaultdict(list); counts=collections.Counter()
    for line in path.open():
        if line.startswith('#'):continue
        f=line.split()
        if len(f)<23 or f[6] not in ('tcp','udp'):continue
        try:
            key=canonical(socket.inet_aton(f[2]),f[3],socket.inet_aton(f[4]),f[5],6 if f[6]=='tcp' else 17)
        except OSError:continue
        start=float(f[0]); end=start+(0 if f[8]=='-' else float(f[8])); y=0 if f[21].lower()=='benign' else 1
        if f[21].lower() not in ('benign','malicious'):counts['unknown_label']+=1;continue
        labels[key].append((start,end,y));counts[f[21]]+=1
    for key,rows in labels.items():
        rows.sort();labels[key]=([r[0] for r in rows],rows)
    return labels,dict(counts)

def label_at(info,t):
    if info is None:return -1
    starts,rows=info;j=bisect.bisect_right(starts,t+1e-6)-1
    found=set()
    # Adjacent Zeek sessions can overlap by a timestamp rounding unit.
    for i in range(max(0,j-2),min(len(rows),j+2)):
        a,b,y=rows[i]
        if a-1e-6<=t<=b+1e-6:found.add(y)
    return next(iter(found)) if len(found)==1 else -1

def prepare_capture(folder):
    clock=time.perf_counter(); out=ROOT/'data/packets'/ (folder.name+'.npz');out.parent.mkdir(exist_ok=True)
    if out.exists():return json.loads(out.with_suffix('.json').read_text())
    labels,label_counts=read_labels(folder/'conn.log.labeled')
    pcap=next(folder.glob('*.pcap')); totals=collections.Counter(); keys={}; rows=[];t0=None
    with pcap.open('rb') as f:
        reader=dpkt.pcap.Reader(f)
        if reader.datalink()!=1:raise ValueError('Expected Ethernet PCAP')
        for index,(ts,buf) in enumerate(reader):
            totals['raw_packets']+=1;ts=float(ts)
            if t0 is None:t0=ts
            try:
                ip=dpkt.ethernet.Ethernet(buf).data
                if not isinstance(ip,dpkt.ip.IP) or ip.p not in (6,17):totals['non_ipv4_tcp_udp']+=1;continue
                if ip.mf or ip.offset:totals['fragmented_ipv4']+=1;continue
                l4=ip.data
                if not isinstance(l4,(dpkt.tcp.TCP,dpkt.udp.UDP)):totals['truncated_transport']+=1;continue
                k=canonical(ip.src,l4.sport,ip.dst,l4.dport,ip.p)
                y=label_at(labels.get(k),ts)
                if k not in keys:keys[k]=len(keys)
                direction=int((ip.src,l4.sport)!=k[0]);flags=int(l4.flags) if ip.p==6 else 0
                rows.append((ts-t0,keys[k],int(ip.len),direction,flags,int(ip.p),y,index))
                totals['matched_packets' if y>=0 else 'unmatched_packets']+=1
            except (dpkt.UnpackError,ValueError,IndexError,AttributeError):totals['parse_errors']+=1
    a=np.asarray(rows,dtype=np.float64); del rows,labels
    # Windows are observation-independent: every transformation retains their identity.
    block=np.floor(a[:,0]/600).astype(np.int64)
    pairs=np.stack([a[:,1].astype(np.int64),block],axis=1)
    _,uid=np.unique(pairs,axis=0,return_inverse=True)
    order=np.lexsort((a[:,7],a[:,0],uid));a=a[order];uid=uid[order];block=block[order]
    begins=np.r_[0,np.flatnonzero(np.diff(uid))+1];ends=np.r_[begins[1:],len(a)]
    low=np.minimum.reduceat(a[:,6],begins).astype(np.int8);high=np.maximum.reduceat(a[:,6],begins).astype(np.int8)
    good=low>=0
    totals['all_communication_windows']=len(begins);totals['unmatched_windows']=int((~good).sum())
    totals['mixed_labeled_windows']=int(((low==0)&(high==1)).sum())
    select=np.repeat(good,ends-begins)
    unit=np.repeat(np.cumsum(good)-1,ends-begins)[select].astype(np.int32)
    a=a[select]; begin=np.r_[0,np.flatnonzero(np.diff(unit))+1];end=np.r_[begin[1:],len(a)]
    # Orient each unit by its first unsampled packet. This orientation is used only
    # to preserve bidirectionality; exporter reorients by its first retained packet.
    arrays=dict(t=a[:,0].astype(np.float64),length=a[:,2].astype(np.float32),direction=a[:,3].astype(np.int8),flags=a[:,4].astype(np.uint8),proto=a[:,5].astype(np.uint8),packet_index=a[:,7].astype(np.int64),unit=unit,begin=begin.astype(np.int64),end=end.astype(np.int64),y=high[good],block=np.floor(a[begin,0]/600).astype(np.int32),tuple_group=a[begin,1].astype(np.int64),unit_start=a[begin,0])
    np.savez_compressed(out,**arrays)
    summary={'capture':folder.name,**dict(totals),'eligible_windows':len(begin),'benign_windows':int((high[good]==0).sum()),'attack_windows':int((high[good]==1).sum()),'retained_packets':len(a),'duration_seconds':float(a[:,0].max()) if len(a) else 0,'label_records':label_counts,'seconds':time.perf_counter()-clock}
    out.with_suffix('.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True);return summary
if __name__=='__main__':
    folders=sorted(p for p in (ROOT/'data/raw').iterdir() if (p/'conn.log.labeled').exists() and list(p.glob('*.pcap')))
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:summary=list(pool.map(prepare_capture,folders))
    (ROOT/'results/capture_audit.json').write_text(json.dumps(summary,indent=2))
