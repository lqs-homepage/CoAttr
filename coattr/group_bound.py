# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["numpy==2.2.6"]
# ///
"""Finite four-coset trace certificate, enumerating degree counts rather than all source subsets."""
import csv
import json
from pathlib import Path
import numpy as np

from designs import budget,native_alias,signed_population,trace_inflation

HERE=Path(__file__).resolve().parent


def factor(n,m,g):
    width=m//g
    sizes=[(j+1)*n//g-j*n//g for j in range(g)]
    axes=np.indices([s+1 for s in sizes],dtype=np.int16)
    s=axes.sum(axis=0)
    tag=np.zeros(s.shape,dtype=np.int16)
    for j,u in enumerate(axes):
        tag ^= j*(u%2)
    kval=np.array([float(signed_population(width,k)) for k in range(width+1)])
    po=np.array([float(native_alias(m,k)) if k<=m else 0.0 for k in range(n+2)])
    base=s*po[np.maximum(s-1,0)]+(n-s)*po[s+1]
    val=np.zeros(s.shape)
    for j,u in enumerate(axes):
        product=np.ones(s.shape)
        for k,v in enumerate(axes):
            if k!=j:
                product*=kval[v]
        minus=(1+(width-1)*product*kval[np.maximum(0,u-1)])/width
        plus=(1+(width-1)*product*kval[np.minimum(width,u+1)])/width
        val+=np.where(tag==j,u*minus+(sizes[j]-u)*plus,0)
    mask=(s>=3)&(s%2==1)
    if np.any(mask&(base<1e-12)&(val>1e-12)):
        return float('inf'),None
    ratio=np.divide(val,base,out=np.zeros(s.shape),where=mask&(base>1e-12))
    arg=np.unravel_index(ratio.argmax(),s.shape)
    return float(ratio[arg]),list(map(int,arg))


def factor_dp(n,m,g):
    """Maximize the exact trace ratio in polynomially many degree/xor states."""
    h=m//g
    sizes=[(j+1)*n//g-j*n//g for j in range(g)]
    k=np.asarray([float(signed_population(h,v)) for v in range(h+1)])
    po=np.asarray([float(native_alias(m,v)) if v<=m else 0.0 for v in range(n+2)])
    worst=0.
    tags=np.arange(g)
    for eligible,nh in enumerate(sizes):
        lo=np.full((1,g),np.inf);hi=np.full((1,g),-np.inf)
        lo[0,0]=hi[0,0]=1.
        for tag,ng in enumerate(sizes):
            if tag==eligible:
                continue
            span=lo.shape[0]
            nextlo=np.full((span+ng,g),np.inf);nexthi=np.full((span+ng,g),-np.inf)
            valid=np.isfinite(lo);low=np.where(valid,lo,0.);high=np.where(valid,hi,0.)
            for u in range(ng+1):
                source=tags^(tag if u%2 else 0)
                x=low[:,source]*k[u];y=high[:,source]*k[u]
                sl=slice(u,u+span)
                nextlo[sl]=np.minimum(nextlo[sl],np.where(valid[:,source],np.minimum(x,y),np.inf))
                nexthi[sl]=np.maximum(nexthi[sl],np.where(valid[:,source],np.maximum(x,y),-np.inf))
            lo,hi=nextlo,nexthi
        for u in range(nh+1):
            wanted=0 if u%2 else eligible
            coeff=(u*k[u-1] if u else 0.)+((nh-u)*k[u+1] if u<nh else 0.)
            for t in range(lo.shape[0]):
                s=t+u
                if s<3 or s%2==0 or not np.isfinite(lo[t,wanted]):
                    continue
                product=hi[t,wanted] if coeff>=0 else lo[t,wanted]
                new=(nh+(h-1)*coeff*product)/h
                old=s*po[s-1]+(n-s)*po[s+1]
                if old>1e-12:
                    worst=max(worst,new/old)
                elif new>1e-12:
                    return float('inf')
    return worst


if __name__=='__main__':
    inputs=json.loads((HERE/'inputs-confirm.json').read_text())
    out=[]
    for rec in inputs:
        n=len(rec['segments']);m=budget(n)//2
        value,where=factor(n,m,4)
        assert abs(factor_dp(n,m,4)-value)<1e-10
        two,_=factor(n,m,2)
        assert abs(trace_inflation(n,m,n//2)-two)<1e-11
        out.append(dict(doc=rec['dataset']+'-'+rec['doc_id'],n=n,B=2*m,two=two,four=value,
                        degree_counts_at_maximum=','.join(map(str,where)) if where else ''))
    with (HERE/'four-group-bound.csv').open('w',newline='') as fp:
        w=csv.DictWriter(fp,fieldnames=list(out[0]),lineterminator='\n');w.writeheader();w.writerows(out)
