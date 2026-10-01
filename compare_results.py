"""Compare generated numerical tables with packaged reference outputs."""
import pathlib,numpy as np,pandas as pd,sys,json
R=pathlib.Path(__file__).resolve().parent;rows=[];failed=False
for expected in sorted((R/'results/expected').rglob('*.csv')):
 relative=expected.relative_to(R/'results/expected');actual=R/'results'/relative
 if not actual.exists():rows.append({'file':str(relative),'status':'not generated'});failed=True;continue
 a=pd.read_csv(expected);b=pd.read_csv(actual)
 if a.shape!=b.shape or list(a.columns)!=list(b.columns):rows.append({'file':str(relative),'status':'shape mismatch'});failed=True;continue
 exact=True;maxdiff=0.;bad=[]
 for col in a:
  if col=='seconds':continue
  if pd.api.types.is_numeric_dtype(a[col]):
   x=a[col].to_numpy();y=b[col].to_numpy();eq=np.isclose(x,y,rtol=0,atol=0,equal_nan=True);exact=exact and eq.all()
   finite=np.isfinite(x)&np.isfinite(y)
   if finite.any():maxdiff=max(maxdiff,float(np.max(np.abs(x[finite]-y[finite]))))
   tolerance=1e-8 if col=='threshold' else 1e-12
   if not np.allclose(x,y,rtol=0,atol=tolerance,equal_nan=True):bad.append(col)
  elif not a[col].equals(b[col]):bad.append(col)
 status='exact' if exact and not bad else ('within tolerance' if not bad else 'difference')
 rows.append({'file':str(relative),'status':status,'maximum_numeric_difference':maxdiff,'different_columns':bad});failed=failed or bool(bad)
print(json.dumps(rows,indent=2));sys.exit(1 if failed else 0)
