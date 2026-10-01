"""Verify the packaged packet-header arrays and reference numerical results."""
import hashlib,pathlib,sys
R=pathlib.Path(__file__).resolve().parent;bad=[];count=0
for line in (R/'MANIFEST.sha256').read_text().splitlines():
 digest,name=line.split('  ',1);p=R/name
 if not p.is_file():bad.append(name);continue
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 if h.hexdigest()!=digest:bad.append(name)
 count+=1
print('Verified',count,'files; mismatches:',bad);sys.exit(bool(bad))
