"""A3-VLA independent synthetic CPU contracts; not official inference."""
from math import exp, log1p, isclose

def ranges(k,w,n):
    if w<=0 or k-2*w<0 or k+w>=n:
        return None
    return list(range(k-2*w,k-w)),list(range(k,k+w))
assert ranges(4,2,7)==([0,1],[4,5])
assert ranges(3,2,7) is None and ranges(4,2,6) is None
print("PASS window boundaries")
assert 65/256>.25 and not 64/256>.25
print("PASS strict patch coverage")
def softmax(x):
    m=max(x)
    a=[exp(v-m) for v in x]
    return [v/sum(a) for v in a]
p=softmax([0,0,0,0])
assert isclose(sum(p[:2]),.5)
assert isclose(sum(softmax([0,0])),1)
print("PASS absolute versus conditional mass")
def loss(x,eta=.8):
    return max(0,eta-sum(softmax(x)[:1]))**2
x=[.1,.2,-.2]
p=softmax(x)
r=p[0]
for j in range(3):
    a=x[:];b=x[:];a[j]+=1e-5;b[j]-=1e-5
    num=(loss(a)-loss(b))/2e-5
    analytic=-2*(.8-r)*p[j]*((j==0)-r)
    assert isclose(num,analytic,rel_tol=1e-6,abs_tol=1e-8)
assert loss([10,0,0])==0
print("PASS hinge finite-difference gradient")
def contrast(pos,neg): return log1p(exp(neg-pos))
assert contrast(.8,.1)<contrast(.1,.8)
print("PASS contrast ordering")
def mean_valid(v,m): return sum(a*b for a,b in zip(v,m))/max(1,sum(m))
assert mean_valid([2,100,4],[1,0,1])==3
assert mean_valid([2],[0])==0
print("PASS valid-only loss denominator")
print("PASS ALL: A3-VLA synthetic contracts")
