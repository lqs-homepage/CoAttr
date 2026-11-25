"""Input-only prefill cost and a Fourier-risk bound for ordered CUBE label splits."""
import math
from collections import defaultdict

from designs import budget,trace_inflation


def history_counts(n,m,cut):
    # t is the affine dimension after the currently exposed column labels.
    distribution={0:1.0}
    counts=[1.0]
    for used in range(1,n):
        nxt=defaultdict(float)
        for dim,mass in distribution.items():
            if cut is not None and used==cut:
                nxt[dim+1]+=mass
                continue
            if cut is None:
                remaining=m-used
                inside=(1<<dim)-used
            elif used<cut:
                remaining=m//2-used
                inside=(1<<dim)-used
            else:
                late=used-cut
                remaining=m//2-late
                inside=(1<<(dim-1))-late
            same=inside/remaining
            assert -1e-12<=same<=1+1e-12
            if same>0:
                nxt[dim]+=mass*same
            if same<1:
                nxt[dim+1]+=mass*(1-same)
        distribution=nxt
        counts.append(sum((1<<t)*p for t,p in distribution.items()))
    return counts


def group_history_counts(n,m,g):
    h=m//g
    tags=[j for j in range(g) for _ in range((j+1)*n//g-j*n//g)]
    profile={0:1.};counts=[1.];projected=0;used=[0]*g;used[0]=1
    for tag in tags[1:]:
        increase=tag>=(1<<projected)
        nxt=defaultdict(float)
        for dim,mass in profile.items():
            same=0. if increase else ((1<<(dim-projected))-used[tag])/(h-used[tag])
            assert -1e-12<=same<=1+1e-12
            if same>0:nxt[dim]+=mass*same
            if same<1:nxt[dim+1]+=mass*(1-same)
        projected+=int(increase)
        used[tag]+=1;profile=nxt
        counts.append(sum((1<<v)*p for v,p in profile.items()))
    return counts


def predicted_cost(doc,answers,qs,m,cut,config,groups=None):
    count=history_counts(doc.n,m,cut) if groups is None else group_history_counts(doc.n,m,groups)
    # Questions preceding the document can share its prefill only when their headers agree.
    groups=defaultdict(list)
    for q in qs:
        groups[tuple(doc.sides[q][2])].append(q)
    tokens=pairs=0.0
    masks=2*count[-1]
    for header_tokens,group in groups.items():
        header=len(header_tokens)
        tails=set()
        for q in group:
            suffix=doc.sides[q][3]+answers[q]['tokens']
            tails.update(tuple(suffix[:i]) for i in range(1,len(suffix)+1))
        prefix=0.0
        for rows,length in zip(count,map(len,doc.parts),strict=True):
            tokens+=rows*length
            pairs+=rows*(length*(header+prefix/2)+length*(length+1)/2)
            prefix+=length
        tokens+=masks*len(tails)
        pairs+=masks*(sum(map(len,tails))+len(tails)*(header+prefix/2))
    hidden=config['hidden_size']
    qdim=config['num_attention_heads']*config['head_dim']
    kdim=config['num_key_value_heads']*config['head_dim']
    linear=2*hidden*(qdim+kdim)+3*hidden*config['intermediate_size']
    layers=config['num_hidden_layers']
    read_positions=sum(1+len(answers[q]['tokens']) for q in qs)
    work=layers*(linear*tokens+2*qdim*pairs)+masks*read_positions*hidden*config['vocab_size']
    return dict(tokens=tokens,pairs=pairs,distinct=masks,work=work)


def options(doc,answers,qs,config,size=None,protect=True):
    n=doc.n
    m=(size or budget(n))//2
    low=max(1,n-m//2+1)
    high=min(n-1,m//2-1)
    cuts=set()
    if low<=high:
        cuts.update(max(low,min(high,round(p*n))) for p in (.40,.45,.50,.55,.60))
        cuts.update((low,high))
    plans=[]
    # Ordinary CUBE is always available. Equal scores break in its favor.
    for cut in [None,*sorted(cuts)]:
        cost=predicted_cost(doc,answers,qs,m,cut,config)
        gamma=1.0 if cut is None or not protect else trace_inflation(n,m,cut)
        plans.append(dict(cut=cut,gamma=gamma,**cost))
    return plans


def choose(plans,protect):
    return min(plans,key=lambda p:p['work']*(p['gamma'] if protect else 1.0))['cut']
