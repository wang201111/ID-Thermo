"""Independent no-Torch/no-pandas execution and full182 parity check."""
import sys,builtins,json,csv
from pathlib import Path
_real_import=builtins.__import__
def guarded_import(name,*args,**kwargs):
 if name.split('.')[0] in {'torch','pandas'}:raise RuntimeError('Forbidden dependency: '+name)
 return _real_import(name,*args,**kwargs)
builtins.__import__=guarded_import
from forward import predict_ray
root=Path(__file__).resolve().parent;errors=[];rounded=[];phases=0;rows=[]
with (root/'verification_182.csv').open(newline='',encoding='utf-8') as f:
 for r in csv.DictReader(f):
  q=[float(r['q_'+s]) for s in ['Na','K','Mg','Cl','SO4']]
  a=predict_ray(q,float(r['temperature_C']),r['system'],allow_unbalanced=True,allow_mass_imbalance=True);b=predict_ray(q,float(r['temperature_C']),r['system'],True,allow_unbalanced=True,allow_mass_imbalance=True)
  assert a['root_found'] and b['root_found'];errors.append(abs(a['salinity_wt_pct']-float(r['expected_S'])));rounded.append(abs(a['salinity_wt_pct']-b['salinity_wt_pct']));phases+=a['controlling_phase']==r['expected_phase'];rows.append({'record_id':r['record_id'],'full_precision':a,'rounded_Table4':b})
assert len(errors)==182 and max(errors)<1e-6 and phases==182
assert not any(k in sys.modules for k in ['torch','pandas'])
result={'status':'PASS','n':182,'max_Torch_reference_difference':max(errors),'controlling_phase_matches':phases,'max_Table4_rounding_difference_wt_pct':max(rounded),'mean_Table4_rounding_difference_wt_pct':sum(rounded)/182,'torch_or_pandas_imported':False}
(root/'verification_result.json').write_text(json.dumps(result,indent=2),encoding='utf-8');(root/'verification_predictions.json').write_text(json.dumps(rows,indent=2),encoding='utf-8');print(result)
