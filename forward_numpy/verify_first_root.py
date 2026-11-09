"""Regression for a real missed early-root training case and its perturbations."""
from pathlib import Path
import json,tempfile
import numpy as np
from forward import predict_ray
root=Path(__file__).resolve().parent;case=json.loads((root/'first_root_regression_case.json').read_text(encoding='utf-8'));base=json.loads((root/'parameters.json').read_text(encoding='utf-8'));checks=[]
with tempfile.TemporaryDirectory() as tmp:
 for variant,sign in [('base',0),('minus',-1),('plus',1)]:
  x=np.array(case['x'])+sign*1e-6*np.array(case['direction'])
  for r,v in zip(base['corrections']['theta'],x[:4]):r['value']=float(v)
  for r,v in zip(base['corrections']['psi'],x[4:]):r['value']=float(v)
  p=Path(tmp)/'parameters.json';p.write_text(json.dumps(base),encoding='utf-8')
  kw=dict(allow_unbalanced=True,allow_mass_imbalance=True,parameter_file=p)
  a=predict_ray(case['q'],case['temperature_C'],'Na-K-Cl-SO4',root_mode='legacy',**kw)
  b=predict_ray(case['q'],case['temperature_C'],'Na-K-Cl-SO4',root_mode='refined',**kw)
  source=next(r for r in case['records'] if r['variant']==variant);expected=source['all_positive_crossings_26_32'][0]['root']
  assert abs(b['salinity_wt_pct']-expected)<1e-7 and b['controlling_phase']=='Mirabilite'
  checks.append({'variant':variant,'legacy_NumPy_S':a['salinity_wt_pct'],'archived_Torch54_S':source['ordinary_root'],'refined_S':b['salinity_wt_pct'],'independent_expected_first_S':expected,'max_error':abs(b['salinity_wt_pct']-expected)})
assert checks[0]['archived_Torch54_S']-checks[0]['refined_S']>.7
result={'status':'PASS','cases':checks,'fixture':'real training-only thirteen-parameter stress case; no quaternary selection','parameter_point':'diagnostic candidate, NOT the delivered nine-parameter table'}
(root/'first_root_regression_result.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(result)
