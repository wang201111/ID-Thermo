"""Phase-resolved first-crossing search with local peak refinement.

This numerical search is independently checked at two grid spacings. It is not
a mathematical certificate against arbitrarily oscillatory user models.
"""
import numpy as np
from scipy.optimize import brentq,minimize_scalar
import pitzer_core as core

def solve_S(m_exp,S_exp,TK,db,mix,phases,grid_step=.1,tol=1e-9):
 if not phases:return np.nan,None,{'reason':'empty_candidate_set'}
 if not 0<S_exp<100 or grid_step<=0:raise ValueError('invalid reference salinity or grid step')
 base=(100-S_exp)/S_exp
 def mol(s):return {k:v*base*s/(100-s) for k,v in m_exp.items() if v>0}
 def all_si(s):
  m=mol(s);lng,lnaw,_,_=core.activities(m,TK,db,mix)
  return np.array([core.si_phase(m,TK,db,mix,p,cache=(lng,lnaw)) for p in phases])
 old,oldphase=core.solve_S(m_exp,S_exp,TK,db,mix,phases,tol=tol)
 lo=.02
 initial=all_si(lo)
 if np.max(initial)>0:return np.nan,None,{'reason':'already_supersaturated_at_lower_bound','lower_bound':lo}
 hi=min(96.,old+grid_step) if np.isfinite(old) else 96.
 grid=np.r_[np.arange(lo,hi,grid_step),hi];values=np.array([all_si(float(s)) for s in grid]);candidates=[];peak_checks=0;peak_rescues=0
 for j,p in enumerate(phases):
  y=values[:,j]
  if not np.isfinite(y).any():continue
  def single(s):return core.si_phase(mol(s),TK,db,mix,p)
  positive=np.flatnonzero(y>=0)
  if len(positive):
   k=int(positive[0])
   if k==0:continue
   root=brentq(single,float(grid[k-1]),float(grid[k]),xtol=tol,rtol=1e-12,maxiter=200);candidates.append((float(root),p))
  # Check every resolved local maximum before the current earliest candidate.
  peaks=np.flatnonzero((y[1:-1]>=y[:-2])&(y[1:-1]>=y[2:])&np.isfinite(y[1:-1]))+1
  for k in peaks:
   earliest=min([r for r,_ in candidates],default=hi)
   if grid[k-1]>=earliest:continue
   # A sign change already brackets this local peak; its left root was tested.
   if y[k]>=0:continue
   peak_checks+=1
   sol=minimize_scalar(lambda s:-single(float(s)),bounds=(float(grid[k-1]),float(grid[k+1])),method='bounded',options={'xatol':1e-12,'maxiter':120})
   peak=float(sol.x);height=single(peak)
   if not sol.success:raise RuntimeError('Local SI maximum refinement failed for '+p)
   if height>1e-12 and single(float(grid[k-1]))<=0:
    root=brentq(single,float(grid[k-1]),peak,xtol=tol,rtol=1e-12,maxiter=200);candidates.append((float(root),p));peak_rescues+=1
 if not candidates:return np.nan,None,{'reason':'no_crossing_within_search_domain','grid_step':grid_step,'peak_checks':peak_checks,'peak_rescues':peak_rescues}
 root,phase=min(candidates,key=lambda z:z[0]);actual=all_si(root);phase=phases[int(np.argmax(actual))]
 return root,phase,{'reason':'crossing_found','grid_step':grid_step,'peak_checks':peak_checks,'peak_rescues':peak_rescues,'legacy_root':float(old) if np.isfinite(old) else None,'legacy_phase':oldphase,'earlier_than_legacy':bool(np.isfinite(old) and root<old-1e-6),'max_SI_residual':float(np.max(actual))}
