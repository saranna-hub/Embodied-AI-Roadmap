#!/usr/bin/env python3
"""Independent synthetic CPU contracts; not official checkpoint inference."""
import math

def close(a,b,tol=1e-7):
    assert abs(a-b)<tol,(a,b)

def soft(errors,t):
    tau=1-t
    logits=[-(t/tau)**2*e for e in errors]
    m=max(logits)
    w=[math.exp(z-m) for z in logits]
    s=sum(w)
    loss=-(tau**3/t)*(m+math.log(s)-math.log(len(errors)))
    return loss,[v/s for v in w]

e=[1.3,.8,.12,.55,1.1]
assert soft(e,.2)[1][2]<soft(e,.5)[1][2]<soft(e,.7)[1][2]
for t in [.2,.5,.7]:
    _,q=soft(e,t)
    close(sum(q),1.)
    for h in range(len(e)):
        plus=e.copy();minus=e.copy()
        plus[h]+=1e-5;minus[h]-=1e-5
        fd=(soft(plus,t)[0]-soft(minus,t)[0])/2e-5
        close(fd,t*(1-t)*q[h])
print("PASS soft-time finite difference and concentration")

# Complete pose must match at a common execution index.
p=[0.,4.]; rot=[4.,0.]
assert min(p)+min(rot)==0
assert min(a+b for a,b in zip(p,rot))==4
# E[(w-target)^2] differs from squared mean error for binary gripper.
mean=.5;second=.5;target=.5
close(second-2*target*mean+target**2,.25)
print("PASS shared arrival and gripper moments")

# Current expert Jacobian matters. v(x)=2x+1, m=x+(1-t)v.
t=.6;x=.2;target=.8
def loss(x):
    return .5*(x+(1-t)*(2*x+1)-target)**2
m=x+(1-t)*(2*x+1)
exact=(m-target)*(1+(1-t)*2)
close((loss(x+1e-5)-loss(x-1e-5))/2e-5,exact)
assert abs(exact-(m-target))>1e-3
# Convert constant velocity from noise time to generation time.
a=2.;noise=-1.;vg=a-noise;vn=-vg
xp=noise;xn=noise
for _ in range(10):
    xp+=.1*vg;xn-=.1*vn
close(xp,a);close(xn,a)
print("PASS expert Jacobian and flow-time sign")

# Delta tracking command pose is not cumulative measured state.
p=0.;cmd=[]
for _ in range(3):
    cmd.append(p+.01);p+=.25*.01
for got,want in zip(cmd,[.01,.0125,.015]):close(got,want)
assert abs(cmd[-1]-.03)>.01
print("PASS delta tracking command versus measurement")

def rms(v,mask):
    n=sum(mask);assert n
    return math.sqrt(sum(x*x for x,m in zip(v,mask) if m)/n)
g=[3.,4.,999.];v=[1.,1.,999.];mask=[True,True,False]
G,V=rms(g,mask),rms(v,mask)
gain,cap=2.,.25
alpha=min(gain,cap*V/(G+1e-8))
b=[alpha*x if m else 0. for x,m in zip(g,mask)]
assert rms(b,mask)<=cap*V+1e-8 and b[-1]==0
demand=gain*G/(V+1e-8)
confidence=math.exp(-max(.1,demand))
assert confidence<math.exp(-1) and rms(b,mask)<1
print("PASS masked cap and unclipped demand rejection")

# Teaching lifecycle oracle, not production runtime.
def executable(proposal, obs, version, consumed=False):
    return (proposal["status"]=="accepted" and
            proposal["obs"]==obs and proposal["version"]==version and
            not consumed)
p={"status":"accepted","obs":"o1","version":7}
assert executable(p,"o1",7)
assert not executable(p,"o1",8)
assert not executable(p,"o2",7)
assert not executable(p,"o1",7,True)
assert not executable(dict(p,status="rejected"),"o1",7)
print("PASS proposal version and single-use gate")
print("PASS ALL: InteractVLA synthetic contracts")
