#!/usr/bin/env python3
"""CPU-only SPAR synthetic data/gradient contracts. Not official model code."""
import math
def future_offsets(h, robot_fps, stream_fps, n=1):
    if h <= 0 or min(robot_fps,stream_fps,n)<=0: raise ValueError("invalid")
    a=[math.floor((h/robot_fps)*stream_fps*k/n+.5+1e-9) for k in range(1,n+1)]
    if a[0]<=0 or len(set(a))!=n: raise ValueError("future grid unrepresentable")
    return a
def longest_bad(seq):
    run=worst=0
    for x in seq:
        run=0 if x else run+1
        worst=max(worst,run)
    return worst
def gate1(flow):
    return bool(flow) and sum(flow)/len(flow)>=.85 and sum(flow)>=3 and longest_bad(flow)<=2
def gate2(hand):
    return bool(hand) and sum(hand)/len(hand)>=.10 and sum(hand)>=1 and longest_bad(hand)<=5
def test_timing():
    assert future_offsets(8,15,15)==[8]
    assert future_offsets(8,15,30)==[16]
    assert future_offsets(8,15,15,4)==[2,4,6,8]
    assert future_offsets(8,15,30,4)==[4,8,12,16]
    assert future_offsets(8,20,20)==[8]
    try: future_offsets(1,30,5,4)
    except ValueError: pass
    else: raise AssertionError("missing negative control")
    print("PASS temporal physical horizon 15Hz to 30Hz")
def test_gates():
    assert not gate1([]) and gate1([True]*18+[False]*2)
    assert not gate1([True]*3+[False]*3)
    assert gate2([False]*4+[True]+[False]*5)
    assert not gate2([]) and not gate2([False]*6+[True]*4)
    area=(.4-.1)*(.5-.12)
    assert .02<=area<=.5
    print("PASS Gate1 flow and Gate2 hand including invalid runs")
def test_flow():
    clean=[.8,-.5,1.2]; noise=[-.2,.5,-.3]
    v=[x-e for x,e in zip(clean,noise)]
    for s in (0,.3,.5,1):
        x=[(1-s)*e+s*a for a,e in zip(clean,noise)]
        assert all(abs(a-(e+s*u))<1e-12 for a,e,u in zip(x,noise,v))
    assert abs(((1-.3)*(-.2)+.3*.8)-.1)<1e-12
    print("PASS linear Flow Matching velocity and hand example")
def test_routes():
    robot={"current":"rgb","action_gt":[1,2],"future":"target"}
    ego={"current":"rgb","future":"target"}
    route=lambda sample: {"gen_loss","action_loss"} if "action_gt" in sample else {"gen_loss"}
    assert route(robot)=={"gen_loss","action_loss"} and route(ego)=={"gen_loss"}
    assert sum([1,1])/2+sum([3,3,3,3])/4==4  # two separately averaged streams
    infer={"current":robot["current"]}
    assert "future" not in infer and "action_gt" not in infer
    print("PASS source-bound action/gen loss and no future inference")
if __name__=="__main__":
    test_timing(); test_gates(); test_flow(); test_routes()
    print("PASS ALL: EgoGuidance SPAR CPU synthetic contracts")
