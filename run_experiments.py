"""Run the pinned, offline numerical experiment from packaged packet headers."""
import argparse,pathlib,subprocess,sys,shutil
ROOT=pathlib.Path(__file__).resolve().parent

def main():
 p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=4);p.add_argument('--device',choices=['cpu','mps'],default='cpu');p.add_argument('--fresh',action='store_true');a=p.parse_args()
 if a.workers<1:raise ValueError('workers must be positive')
 results=ROOT/'results';results.mkdir(exist_ok=True)
 if a.fresh:
  shutil.rmtree(ROOT/'data/features',ignore_errors=True)
  for pattern in ['classification_*.csv','grouped_*.csv','grouped_*.json','timing_*.json','drift_*.csv','drift_*.json']:
   for file in results.glob(pattern):file.unlink()
 jobs=[('export.py',['--workers',str(a.workers)]),('evaluate.py',['--workers',str(a.workers)]),('evaluate_grouped.py',['--workers',str(a.workers)]),('drift_control.py',['--device',a.device]),('drift.py',['--device',a.device]),('summarize.py',[])]
 for name,args in jobs:
  print('Running',name,flush=True);subprocess.run([sys.executable,str(ROOT/'src'/name),*args],cwd=ROOT,check=True)
 print('Numerical results written to',results)
if __name__=='__main__':main()
