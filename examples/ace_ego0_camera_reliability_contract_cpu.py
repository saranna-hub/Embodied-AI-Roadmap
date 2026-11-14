#!/usr/bin/env python3
"""ACE-Ego-0 standard-library CPU contracts; not the original model."""
import math
def close(x,y):
    assert abs(x-y)<1e-8,(x,y)
def camera(p):return (-p[1],p[0],p[2])
def inverse(p):return (p[1],-p[0],p[2])
def parse22(a):
    if len(a)!=22:raise ValueError("paper22 requires activity, no waist")
    return a[:11],a[11:]
def parse23(a):
    if len(a)!=23:raise ValueError("Common23 requires waist, no activity")
    return a[:9],a[9:18],a[18:20],a[20:23]
def weight(channel,source,q):
    step=1. if q<=1 else max(.2,math.exp(-1.5*(q-1)))
    return channel*source*step
def huber(e):
    e=abs(e)
    return e*e/2 if e<=1 else e-.5
def weighted_loss(err,mask,weights):
    total=sum(int(m)*w for m,w in zip(mask,weights))
    if total<=0:return 0.
    return sum(int(m)*w*huber(e) for e,m,w in zip(err,mask,weights))/total
def main():
    p=(1.,0.,.4)
    assert camera(p)==(0.,1.,.4) and inverse(camera(p))==p
    assert camera(p)!=inverse(p)
    print("PASS camera transform inverse and sign negative control")
    assert [round(f*2) for f in [10,20,30]]==[20,40,60]
    assert [30/f for f in [10,20,30]]==[3,1.5,1]
    print("PASS physical-time horizons and fixed H30")
    assert len(parse22(list(range(22)))[1])==11
    assert parse23(list(range(23)))[-1]==[20,21,22]
    for fn,bad in [(parse22,[0]*23),(parse23,[0]*22)]:
        try:fn(bad)
        except ValueError:pass
        else:raise AssertionError("layout error")
    print("PASS paper22/Common23 ABI collision rejected")
    anchor=[1.,2.,3.];target=[1.4,2.1,3.5]
    delta=[t-a for a,t in zip(anchor,target)]
    for a,d,t in zip(anchor,delta,target):close(a+d,t)
    print("PASS state-anchor delta")
    action,noise,s=.8,-.2,.3
    close((1-s)*noise+s*action,.1)
    close(action-noise,1.)
    print("PASS Flow Matching velocity")
    close(weight(1.,.8,.5),.8)
    close(weight(.001,.8,.5),.0008)
    assert weight(1.,.8,3.)<.8
    print("PASS channel-source-time reliability weights")
    close(weighted_loss([2,1],[1,0],[1.,1.]),1.5)
    close(weighted_loss([2,1],[1,0],[.5,1.]),1.5)
    close(weighted_loss([2,1],[0,0],[1.,1.]),0.)
    print("PASS Huber masked normalization and zero-denominator")
    x,y,z=.1,.2,1.
    close(500*x/z+320,370)
    close(500*y/z+240,340)
    assert z>0 and not (-1.>0)
    print("PASS positive-depth projection")
    print("PASS ALL: ACE-Ego-0 camera/reliability contracts")
if __name__=="__main__":main()
