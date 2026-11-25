#!/usr/bin/env python3
"""Independent synthetic contracts, not official TacInno inference."""
from itertools import product
from math import isclose

# Conditional expectation oracle: actions and residual independent.
samples=[(a,z,2*a+z) for a in (-2,0,2) for z in (-1,1)]
for a in (-2,0,2):
    ys=[y for aa,z,y in samples if aa==a]
    assert sum(ys)/len(ys)==2*a
assert sum((y-2*a)*a for a,z,y in samples)==0
assert sum((y-2*a)**2 for a,z,y in samples)==6
print("PASS conditional residual orthogonality")

# Four independent codebooks and integer label range.
assert len(set(product(range(16),repeat=4)))==65536
def valid_code(c): return len(c)==4 and all(isinstance(x,int) and 0<=x<16 for x in c)
assert valid_code((0,2,5,15)) and not valid_code((0,2,5,16))
print("PASS four independent 16-way codes")

def valid_window(t,episode,valid):
    offsets=(-8,-4,0,5,10,15)
    return all(0<=t+i<len(episode) and episode[t+i]==episode[t] and valid[t+i] for i in offsets)
ep=[0]*40+[1]*40
valid=[True]*80
assert valid_window(10,ep,valid) and not valid_window(30,ep,valid)
assert not valid_window(42,ep,valid)
valid[20]=False
assert not valid_window(10,ep,valid)
print("PASS future and history episode isolation")

# Query/key block contract; transitive dependency across multiple layers.
M=[[1,0,0,0,0],[1,1,0,0,0],[1,1,1,0,0],[1,1,1,1,0],[1,1,1,0,1]]
deps=[{i} for i in range(5)]
for _ in range(8):
    deps=[set.union(*(deps[j] for j in range(5) if row[j])) for row in M]
assert 3 not in deps[4] and 4 not in deps[3]
assert 2 in deps[3] and 2 in deps[4]
# Negative control: ordinary monotonic causal mask leaks Q to A.
assert 3 in {j for j in range(5) if j<=4}
print("PASS multilayer auxiliary-query isolation")

q=.1
targets=[.2,.3]
anchor=[v-q for v in targets]
step=[targets[0]-q,targets[1]-targets[0]]
assert isclose(anchor[1],.2) and isclose(step[1],.1)
assert not isclose(sum(anchor),targets[-1]-q)
print("PASS anchored versus incremental action")

def masked_mean(values,mask):
    return sum(v for v,m in zip(values,mask) if m)/max(1,sum(mask))
assert masked_mean([2,8,999],[1,1,0])==5
assert masked_mean([999],[0])==0
print("PASS independent loss validity masks")
print("PASS ALL: TacInno synthetic contracts")
