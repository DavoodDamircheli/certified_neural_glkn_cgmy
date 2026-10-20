#!/usr/bin/env python3
"""Extended-precision (x86 long double) check of the reduced certificate's equality floor (plan 2.3, xfail
test): for the worst validation query (POD-Galerkin r = 48, basis 800, rC cells), compare the online
eta^{X-bar} and the double-precision full-order value against a long-double full-order reference."""
import sys, json, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
from run_reduced_cert import *
res=json.load(open(ROOT / 'results/E2_3_reduced_cert.json')); cp=cell_params(res)
rows=res['check']['rows']; i=int(np.argmax([r['cands']['pod_galerkin']['eta_rel_diff'] for r in rows]))
d=np.load(DATA/f"references_N{N}_NT{NT}_validation30.npz"); m=d['mu'][i]
q=Query(C=m[0],G=m[1],M=m[2],Y=m[3],r=m[4],T=float(d['T'][i]))
e=load_eim(N); rc=load_rc(rc_path(800)); V=np.load(DATA/f"pod_N{N}_NT{NT}_seed0_800.npz")["V"][:,:R]
rom=AffineROM(V,e); fsQ,info=e.affine_free_system(q); forc=free_forcing(fsQ,q.T,NT)
sol=rom.solve(q,rc.alpha0(),NT,ell=info['ell']); al=sol.alpha; cid=cell_of(q)
ld=np.longdouble
Vl=V.astype(ld); Z=Vl@al.astype(ld)
A=fsQ.A.dense().astype(ld)  # A_Q dense from lags (entries exact as stored doubles)
from cgmy_cert.symbols import _dense_banded
M=_dense_banded(fsQ.M_entries,fsQ.n).astype(ld)
dt=ld(sol.dt); bL=sol.bL.astype(ld)
uM=forc.uM.astype(ld); uA=forc.uA.astype(ld)
Rr=M@((Z[:,1:]-Z[:,:-1])/dt)+A@((Z[:,1:]+Z[:,:-1])/2)+np.outer(uM,(bL[1:]-bL[:-1])/dt)+np.outer(uA,(bL[1:]+bL[:-1])/2)
Xb=_dense_banded(xbar_entries(*cp[cid]),fsQ.n).astype(ld)
# banded Cholesky solve in longdouble (bandwidth 2): plain Gaussian elimination on dense (no pivoting, SPD)
n=fsQ.n; Xc=Xb.copy(); B=Rr.copy()
for k in range(n):
    for j in range(k+1,min(k+3,n)):
        f=Xc[j,k]/Xc[k,k]; Xc[j,k:k+3]-=f*Xc[k,k:k+3]; B[j]-=f*B[k]
for k in range(n-1,-1,-1):
    B[k]=(B[k]-Xc[k,k+1:k+3]@B[k+1:k+3])/Xc[k,k]
ref=np.sum(Rr*B,axis=0)
X=np.vstack([(al[:,1:]-al[:,:-1])/sol.dt,((sol.bL[1:]-sol.bL[:-1])/sol.dt)[None]]); Y=np.vstack([0.5*(al[:,1:]+al[:,:-1]),(0.5*(sol.bL[1:]+sol.bL[:-1]))[None]])
u=rc.per_step(cid,rc.theta(q,info['ell']),X,Y)
Rd=cn_residuals(fsQ,V@al,forc); full,_=banded_dual_norms(xbar_entries(*cp[cid]),n,2,Rd)
eta=lambda v: np.sqrt(np.sum(v,dtype=ld))
ev=dict(query_index=i, online_vs_longdouble=float(abs(eta(u)-eta(ref))/eta(ref)),
        full_order_double_vs_longdouble=float(abs(eta(full)-eta(ref))/eta(ref)), longdouble_eps=float(np.finfo(ld).eps),
        script='scripts/check_reduced_cert_longdouble.py')
res['check']['extended_precision_evidence']=ev
open(ROOT / 'results/E2_3_reduced_cert.json','w').write(json.dumps(res, indent=1, default=float))
print(ev)
