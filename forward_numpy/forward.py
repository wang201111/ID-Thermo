"""Portable first-saturation prediction. Requires only NumPy and SciPy."""
from pathlib import Path
import json,csv,argparse,math
import numpy as np
import pitzer_core as core
ROOT=Path(__file__).resolve().parent
IONS=['Na+','K+','Mg+2','Cl-','SO4-2']

def ray_from_salt_weights(weights):
 """Anhydrous salt masses (any common unit) -> ionic q, mol/kg dry salts."""
 stoich={'NaCl':[1,0,0,1,0],'KCl':[0,1,0,1,0],'MgCl2':[0,0,1,2,0],'Na2SO4':[2,0,0,0,1],'K2SO4':[0,2,0,0,1],'MgSO4':[0,0,1,0,1]}
 atomic=np.array([22.989769,39.0983,24.305,35.453,96.06]);q=np.zeros(5);total=0.
 for name,value in weights.items():
  if name not in stoich or not math.isfinite(value) or value<0:raise ValueError('invalid anhydrous salt weight: '+str(name))
  st=np.array(stoich[name]);q+=1000*value*st/(st@atomic);total+=value
 if total<=0:raise ValueError('salt weights must have positive total')
 return q/total

def predict_ray(q,temperature_C,system,rounded=False,allow_unbalanced=False,parameter_file=None,allow_mass_imbalance=False,root_mode='refined'):
 q=np.asarray(q,dtype=float)
 if q.shape!=(5,) or not np.isfinite(q).all() or (q<0).any() or q.sum()<=0:raise ValueError('q requires five finite nonnegative ionic ray coefficients')
 charge_error=float(abs(q@np.array([1,1,2,-1,-2]))/max(1.,q.sum()))
 if charge_error>1e-6 and not allow_unbalanced:raise ValueError('ionic ray is not electroneutral; --allow-unbalanced is only for explicit historical reproduction')
 mass_total=float(q@np.array([22.989769,39.0983,24.305,35.453,96.06]));mass_error=abs(mass_total/1000.-1.)
 if mass_error>1e-3 and not allow_mass_imbalance:raise ValueError('q is not normalized to 1 kg dry salts; --allow-mass-imbalance is only for explicit historical reproduction')
 if not math.isfinite(temperature_C) or not 0<=temperature_C<=100:raise ValueError('supported computational domain is 0 to 100 Celsius; validation range is narrower for some systems')
 parameter_file=Path(parameter_file) if parameter_file else ROOT/'parameters.json'
 data=json.loads(parameter_file.read_text(encoding='utf-8'));db=core.ParamDB(parameter_file);tk=temperature_C+273.15
 if system not in data['systems']:raise ValueError('system must be one of '+','.join(data['systems']))
 allowed={'Na-K-Cl-SO4':{0,1,3,4},'Na-Mg-Cl-SO4':{0,2,3,4},'K-Mg-Cl-SO4':{1,2,3,4}}[system]
 if any(q[i]>1e-14 for i in set(range(5))-allowed):raise ValueError('ion composition conflicts with selected system')
 for p,fit in data['logk_fits'].items():
  if p in db.phase:
   beta=fit['beta'];value=beta[0]+beta[1]/tk+(beta[2]*np.log(tk) if fit['k']==3 else 0)
   db.phase[p]=(db.phase[p][0],float(value))
 theta={};psi={}
 for r in data['corrections']['theta']:
  ids=r['ions'];delta=round(r['value'],4) if rounded else r['value'];theta[frozenset(ids)]=lambda tt,ids=ids,delta=delta:db.theta(*ids,tt)+delta
 for r in data['corrections']['psi']:
  ids=r['ions'];delta=round(r['value'],4) if rounded else r['value'];psi[frozenset(ids)]=lambda tt,ids=ids,delta=delta:db.psi(*ids,tt)+delta
 allp=data['systems'][system];valid=[p for p in allp if p not in data['logk_fits'] or data['logk_fits'][p]['T_min']<=temperature_C<=data['logk_fits'][p]['T_max']]
 family=lambda p:frozenset(core.parse_phase_safe(db.phase[p][0]))-{'H2O'}
 families={family(p) for p in valid};phases=[p for p in allp if p in valid or family(p) not in families]
 mix=core.Mix(theta,psi)
 # S_reference=50 gives base=1, so q directly defines m_i=q_i*S/(100-S).
 if root_mode=='legacy':
  pred,phase=core.solve_S(dict(zip(IONS,q)),50.,tk,db,mix,phases);search={'method':'archived54point_scan'}
 elif root_mode=='refined':
  from first_root import solve_S
  pred,phase,search=solve_S(dict(zip(IONS,q)),50.,tk,db,mix,phases);search['method']='phase_resolved_grid0.1_local_peak_refinement'
 else:raise ValueError('root_mode must be refined or legacy')
 outside=(system=='Na-Mg-Cl-SO4' and temperature_C>55) or (system=='K-Mg-Cl-SO4' and temperature_C!=15)
 return {'salinity_wt_pct':float(pred) if np.isfinite(pred) else None,'controlling_phase':phase,'root_found':bool(np.isfinite(pred)),'candidate_phases':phases,'outside_observed_quaternary_temperature_range':bool(outside),'rounded_Table4':rounded,'relative_charge_error':charge_error,'unbalanced_input':bool(charge_error>1e-6),'ray_mass_g_per_kg_dry_salts':mass_total,'mass_normalization_relative_error':mass_error,'mass_inconsistent_input':bool(mass_error>1e-3),'root_search':search}

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--input',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--rounded',action='store_true');ap.add_argument('--allow-unbalanced',action='store_true');ap.add_argument('--allow-mass-imbalance',action='store_true');ap.add_argument('--parameters',type=Path,default=ROOT/'parameters.json');ap.add_argument('--root-mode',choices=['refined','legacy'],default='refined');args=ap.parse_args()
 if args.output.exists():raise FileExistsError('Refusing to overwrite '+str(args.output))
 rows=[]
 with args.input.open(encoding='utf-8-sig',newline='') as f:
  for row in csv.DictReader(f):
   r=predict_ray([float(row['q_'+s]) for s in ['Na','K','Mg','Cl','SO4']],float(row['temperature_C']),row['system'],args.rounded,args.allow_unbalanced,args.parameters,args.allow_mass_imbalance,args.root_mode)
   rows.append({'record_id':row.get('record_id',''),**r})
 args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(rows,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
 print(f'Predicted {len(rows)} compositions -> {args.output}')
if __name__=='__main__':main()
