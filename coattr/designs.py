"""Binary coalition queries. Bit i denotes keeping source i in its original position."""
import hashlib
import math
import random
from fractions import Fraction
from functools import lru_cache


def seed(*parts):
    return int.from_bytes(hashlib.sha256('/'.join(map(str,parts)).encode()).digest()[:8],'big')


def budget(n):
    return 2 * (1 << n.bit_length())


def cube(n, size, rng, cut=None):
    """CUBE or its ordered two-coset variant. One independent uniform sign shift."""
    m=size//2
    if size!=2*m or m & (m-1) or not 1<=n<m:
        raise ValueError('Invalid CUBE dimension or indexed budget')
    if cut is None:
        labels=rng.sample(range(1,m),n)
    else:
        half=m//2
        if not 1<=cut<n or max(cut,n-cut)>half:
            raise ValueError('Invalid two-coset split')
        labels=rng.sample(range(half),cut)+rng.sample(range(half,m),n-cut)
    return label_rows(labels,m,rng)


def partition_cube(n,size,rng,groups=4):
    """Balanced consecutive source ranges in distinct cosets of a common subspace."""
    m=size//2
    if groups & (groups-1) or m%groups or not groups<=n<m:
        raise ValueError('Invalid coset dimensions')
    width=m//groups
    labels=[]
    for group in range(groups):
        count=(group+1)*n//groups-group*n//groups
        labels.extend(rng.sample(range(group*width,(group+1)*width),count))
    return label_rows(labels,m,rng)


def label_rows(labels,m,rng):
    n=len(labels)
    shift=rng.getrandbits(n)
    all_bits=(1<<n)-1
    rows=[]
    for t in range(m):
        mask=shift
        for i,label in enumerate(labels):
            mask ^= ((label & t).bit_count()%2)<<i
        rows.extend((mask,all_bits^mask))
    return rows


def antithetic(n, size, rng):
    rows=[]
    for _ in range(size//2):
        mask=rng.getrandbits(n)
        rows.extend((mask,((1<<n)-1)^mask))
    return rows


@lru_cache(maxsize=None)
def signed_population(m,s):
    if s%2:
        return Fraction()
    return Fraction((-1)**(s//2)*math.comb(m//2,s//2),math.comb(m,s))


@lru_cache(maxsize=None)
def native_alias(m,s):
    if s%2:
        return Fraction()
    return (1+(m-1)*signed_population(m,s))/m


@lru_cache(maxsize=None)
def split_alias(m,a,b):
    if a%2 or b%2:
        return Fraction()
    half=m//2
    return (1+(half-1)*signed_population(half,a)*signed_population(half,b))/half


@lru_cache(maxsize=None)
def inflation(n,m,cut):
    """Worst alias-risk ratio over all nonconstant even character supports."""
    ratios=[Fraction(1)]
    for a in range(0,cut+1,2):
        for b in range(0,n-cut+1,2):
            if a+b<4:
                continue
            ordinary=native_alias(m,a+b)
            divided=split_alias(m,a,b)
            if not ordinary:
                if divided:
                    return math.inf
            else:
                ratios.append(divided/ordinary)
    return float(max(ratios))


@lru_cache(maxsize=None)
def trace_inflation(n,m,cut):
    """Sharp ratio for the summed singleton MSE over arbitrary odd-response spectra."""
    late=n-cut
    worst=Fraction()
    for a in range(cut+1):
        for b in range(late+1):
            s=a+b
            if s<3 or s%2==0:
                continue
            base=s*native_alias(m,s-1)
            if s<n:
                base+=(n-s)*native_alias(m,s+1)
            divided=Fraction()
            for count,u,v in [(a,a-1,b),(cut-a,a+1,b),(b,a,b-1),(late-b,a,b+1)]:
                if count:
                    divided+=count*split_alias(m,u,v)
            if base:
                worst=max(worst,divided/base)
            elif divided:
                return math.inf
    return float(worst)
