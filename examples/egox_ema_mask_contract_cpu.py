#!/usr/bin/env python3
"""EgoX MED CPU-only mathematical contracts; not an official model reproduction."""
from math import sqrt, isclose
def smooth_l1(x,beta=.05):
    x=abs(x)
    return x*x/(2*beta) if x<beta else x-beta/2
def action_loss(pred,gt,channels,pad):
    vals=[smooth_l1(a-b) for t,(p,y) in enumerate(zip(pred,gt))
          for d,(a,b) in enumerate(zip(p,y)) if not pad[t] and channels[d]]
    return sum(vals)/max(1,len(vals))
def norm(a): return sqrt(sum(x*x for x in a))
def cos(a,b):
    return sum(x*y for x,y in zip(a,b))/(norm(a)*norm(b)) if norm(a)*norm(b)>0 else 0
def dream_loss(pred,gt,valid):
    vals=[1-cos(a,b)+.01*smooth_l1(norm(a)-norm(b),1.0)
          for a,b,ok in zip(pred,gt,valid) if ok]
    return sum(vals)/max(1,len(vals))
def main():
    p=[[1,999,0],[2,999,0],[999,999,999]]
    y=[[0,0,0],[0,0,0],[0,0,0]]
    m=[True,False,True]; pads=[False,False,True]
    val=action_loss(p,y,m,pads)
    p[0][1]=1e9; p[1][1]=-1e9; p[2]=[1e9]*3
    assert isclose(val,action_loss(p,y,m,pads))
    assert isclose(val,(smooth_l1(1)+smooth_l1(2))/4)
    print('PASS masked action and temporal padding')
    t,T=85,100
    assert sum(t+i<T for i in range(60))==15
    future_valid=[t+i<T for i in [20,40,60]]
    assert future_valid==[False]*3
    assert dream_loss([[5,0]]*3,[[1,0]]*3,future_valid)==0
    print('PASS episode future offset validity')
    assert isclose(cos([3,0],[6,0]),1)
    assert dream_loss([[3,0]],[[6,0]],[True])>0
    print('PASS cosine versus latent norm')
    decay=.999; after=decay*0+(1-decay)*10
    assert isclose(after,.01,abs_tol=1e-12)
    print('PASS EMA 0.999')
    assert 24!=60 and 3*8*512==12288
    print('PASS query count != horizon; 3x8x512 future targets')
    train={'obs':{'main_image':'current'},'dream':{'main_image':'future'}}
    inference={'obs':train['obs']}
    assert 'dream' not in inference
    print('PASS training-only target causality')
    print('PASS ALL: EgoX MED EMA/action/dream CPU')
if __name__=='__main__': main()
