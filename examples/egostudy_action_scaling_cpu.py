"""Independent synthetic EgoStudy geometry and sampling checks. No model or dataset."""
import math

def dot(a,b): return sum(x*y for x,y in zip(a,b))
def sub(a,b): return tuple(x-y for x,y in zip(a,b))
def add(a,b): return tuple(x+y for x,y in zip(a,b))
def mul(a,s): return tuple(x*s for x in a)
def T(a): return tuple(zip(*a))
def mm(a,b): return tuple(tuple(dot(r,c) for c in T(b)) for r in a)
def mv(a,b): return tuple(dot(r,b) for r in a)
def flat(a): return tuple(v for r in a for v in r)
def close(a,b): assert all(abs(x-y)<1e-9 for x,y in zip(a,b)) and len(a)==len(b)
def rz(t): return ((math.cos(t),-math.sin(t),0),(math.sin(t),math.cos(t),0),(0,0,1))
def rx(t): return ((1,0,0),(0,math.cos(t),-math.sin(t)),(0,math.sin(t),math.cos(t)))
def unit(v):
    n=math.sqrt(dot(v,v))
    if n<1e-8: raise ValueError("degenerate rotation")
    return mul(v,1/n)
def cross(a,b): return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])
def decode(a,b):
    x=unit(a)
    y=unit(sub(b,mul(x,dot(x,b))))
    return T((x,y,cross(x,y)))
def masked_mse(pred,target,mask):
    count=sum(mask)
    return sum((x-y)**2 for x,y,m in zip(pred,target,mask) if m)/count if count else None

def main():
    assert 2*(1+5)*(3+6)==108
    print("PASS 108D action groups")
    c, w0, wt=rz(.7),rx(.4),mm(rz(.5),rx(.8))
    p0,pt=(.2,.1,.3),(.4,.2,.7)
    d=mv(T(c),sub(pt,p0))
    q=mm(mm(mm(T(c),wt),T(w0)),c)
    close(add(p0,mv(c,d)),pt)
    close(flat(mm(mm(mm(c,q),T(c)),w0)),flat(wt))
    wrong=mm(T(w0),wt)
    assert max(abs(x-y) for x,y in zip(flat(q),flat(wrong)))>.01
    print("PASS anchor-frame rotation and translation roundtrip")
    u=(.04,.02,.1); s=rz(.2)
    for p,w in [(p0,w0),(pt,wt)]:
        f=add(p,mv(w,u)); fr=mm(w,s)
        close(mv(T(w),sub(f,p)),u)
        close(flat(mm(T(w),fr)),flat(s))
    print("PASS wrist-local fingertip invariance")
    a,b,_=T(wt)
    close(flat(decode(a,b)),flat(wt))
    try: decode((1,0,0),(2,0,0))
    except ValueError: pass
    else: raise AssertionError("parallel columns accepted")
    print("PASS 6D rotation degenerate reject")
    fixed=list(range(32)); event=[round(k*100/31) for k in range(32)]
    assert fixed[-1]/20==1.55 and event[-1]/20==5
    assert len(event)==32 and len(set(event))==32
    print("PASS event versus fixed physical duration")
    assert masked_mse([2,1000],[0,0],[1,0])==4
    assert masked_mse([2],[0],[0]) is None
    print("PASS group mask excludes invalid and empty labels")
    for p in (.01,.03,.1,.3,1):
        weights=[math.sqrt(h*p) for h in (100,900)]
        close(tuple(w/sum(weights) for w in weights),(.25,.75))
        close((.75*weights[0]/sum(weights),.75*weights[1]/sum(weights),.2,.05),(.1875,.5625,.2,.05))
    print("PASS fixed robot exposure and square-root mixture")
    print("PASS ALL: EgoStudy independent synthetic contracts")

if __name__=="__main__": main()
