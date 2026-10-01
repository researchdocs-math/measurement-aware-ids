"""Download the pinned, public packet captures and original ground truth."""
import concurrent.futures, hashlib, json, pathlib, time, requests, zipfile, shutil
ROOT=pathlib.Path(__file__).resolve().parents[1]
def fetch(row):
    path=ROOT/row['path']; path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        tmp=path.with_suffix(path.suffix+'.part')
        for attempt in range(4):
            try:
                with requests.get(row['url'],stream=True,timeout=(30,120)) as r:
                    r.raise_for_status()
                    with tmp.open('wb') as f:
                        for b in r.iter_content(1024*1024): f.write(b)
                tmp.replace(path); break
            except Exception:
                if attempt==3: raise
                time.sleep(2*(attempt+1))
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    if row.get('sha256') and row['sha256']!=h.hexdigest():raise ValueError('Checksum mismatch: '+str(path))
    if row.get('unpack_zip'):
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                if member.is_dir(): continue
                target=path.parent/pathlib.Path(member.filename).name
                with archive.open(member) as source, target.open('wb') as dest: shutil.copyfileobj(source,dest)
    result={**row,'sha256':h.hexdigest(),'bytes':path.stat().st_size}
    print(json.dumps(result),flush=True);return result
if __name__=='__main__':
    rows=json.loads((ROOT/'configs/sources.json').read_text())+json.loads((ROOT/'configs/kitsune_sources.json').read_text())
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool: results=list(pool.map(fetch,rows))
    (ROOT/'data/download_manifest.json').write_text(json.dumps(results,indent=2))
