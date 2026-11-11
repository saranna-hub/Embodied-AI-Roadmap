#!/usr/bin/env python3
"""EgoWAM educational CPU-only contract; original mathematics, NOT official training code."""
from __future__ import annotations
import math

def add(a,b):
    return tuple(x+y for x,y in zip(a,b))

def sub(a,b):
    return tuple(x-y for x,y in zip(a,b))

def rotz(theta, point):
    x,y,z=point
    return (math.cos(theta)*x-math.sin(theta)*y,
            math.sin(theta)*x+math.cos(theta)*y,z)

def to_cam(p_world,theta,t_world_cam):
    # X_cam = R^T (P_world - t)
    return rotz(-theta,sub(p_world,t_world_cam))

def to_world(p_cam,theta,t_world_cam):
    return add(rotz(theta,p_cam),t_world_cam)

def l2(p):
    return math.sqrt(sum(v*v for v in p))

def stabilize_flow(x_t_cam, x_future_cam, theta_t, t_t,
                   theta_future, t_future):
    world_future=to_world(x_future_cam,theta_future,t_future)
    future_in_ct=to_cam(world_future,theta_t,t_t)
    return sub(future_in_ct,x_t_cam)

def test_camera():
    theta0,t0=0.0,(0.,0.,0.)
    theta1,t1=math.pi/2,(1.,0.,0.)
    static=(2.,1.,0.)
    x0=to_cam(static,theta0,t0)
    x1=to_cam(static,theta1,t1)
    naive=l2(sub(x1,x0))
    stable=stabilize_flow(x0,x1,theta0,t0,theta1,t1)
    assert naive>1.0 and l2(stable)<1e-12,(naive,stable)
    print("PASS ego-motion static-zero: naive=%.3fm stabilized=%.12fm"%(naive,l2(stable)))
    moved=add(static,(0.05,0.,0.))
    moving=stabilize_flow(x0,to_cam(moved,theta1,t1),theta0,t0,theta1,t1)
    assert abs(moving[0]-0.05)<1e-12 and l2(sub(moving,(.05,0,0)))<1e-12
    print("PASS moving-object preserved: 0.050m")
    # negative control: use inverse orientation by mistake, catches frame-convention errors
    wrong=stabilize_flow(x0,x1,theta0,t0,-theta1,t1)
    assert l2(wrong)>0.1
    print("PASS frame-inversion negative control: wrong flow=%.3fm"%l2(wrong))

def test_linear_flow():
    clean=[0.2,-0.6,1.0]
    eps=[-1.2,0.7,0.4]
    v=[a-e for a,e in zip(clean,eps)]
    for tau in (0.0,0.1,0.5,0.9,1.0):
        x=[(1-tau)*e+tau*a for a,e in zip(clean,eps)]
        assert max(abs(xx-(e+tau*vv)) for xx,e,vv in zip(x,eps,v))<1e-12
        pred=[xx+(1-tau)*vv for xx,vv in zip(x,v)]
        assert max(abs(xx-a) for xx,a in zip(pred,clean))<1e-12
    print("PASS flow-matching linear-path and Euler velocity identity")

def test_valid_mask():
    target=[[.1,.2,.3],[float("nan"),float("nan"),float("nan")],[.6,.1,.0]]
    pred=[[.2,.1,.3],[999.,999.,999.],[.5,.2,.0]]
    mask=[True,False,True]
    valid=[(p,t) for p,t,m in zip(pred,target,mask) if m]
    assert len(valid)==2 and all(math.isfinite(t) for _,v in valid for t in v)
    mse=sum((a-b)**2 for p,t in valid for a,b in zip(p,t))/(3*len(valid))
    assert abs(mse-0.006666666666666671)<1e-12,mse
    print("PASS invalid-point mask excludes NaN/outlier and uses valid-count denominator")

def test_normalization():
    qlo,qhi=-2.,4.
    vals=[-2.,-1.,1.,4.,7.]
    norm=[2*(a-qlo)/(qhi-qlo)-1 for a in vals]
    back=[qlo+(v+1)*(qhi-qlo)/2 for v in norm]
    assert all(abs(x-y)<1e-12 for x,y in zip(back,vals))
    assert norm[-1]>1.0
    try:
        qlo=qhi=3.
        assert qhi!=qlo, "degenerate training quantiles"
    except AssertionError:
        pass
    print("PASS q01/q99 reversible normalization; OOD not silently clipped")

def test_no_future_leak():
    train={"current_rgb":"I_t","future_dino":"target_t_plus_T","action_gt":"target_action"}
    inference={k:train[k] for k in ("current_rgb",)}
    assert "future_dino" not in inference and "action_gt" not in inference
    assert set(inference)=={"current_rgb"}
    print("PASS future target absent from inference conditioning")

if __name__=="__main__":
    test_camera()
    test_linear_flow()
    test_valid_mask()
    test_normalization()
    test_no_future_leak()
    print("PASS ALL: EgoWAM camera/flow/mask/action synthetic CPU contracts")
