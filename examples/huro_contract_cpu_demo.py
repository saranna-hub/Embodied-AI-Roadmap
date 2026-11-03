#!/usr/bin/env python3
"""HuRo 官方数据语义的独立 Python 标准库合成契约，非原论文复现。
Usage: python3 examples/huro_contract_cpu_demo.py
"""
from __future__ import annotations
import math


def matmul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)]
            for i in range(4)]


def xyz(T, p):
    v = list(p) + [1.]
    q = [sum(T[i][j] * v[j] for j in range(4)) for i in range(4)]
    assert abs(q[3]-1.0) < 1.e-9
    return q[:3]


def inverse(T):
    R = [row[:3] for row in T[:3]]
    Rt = [[R[j][i] for j in range(3)] for i in range(3)]
    t = [T[i][3] for i in range(3)]
    return [Rt[i]+[-sum(Rt[i][j]*t[j] for j in range(3))]
            for i in range(3)] + [[0.,0.,0.,1.]]


def close(a,b):
    assert len(a)==len(b) and all(abs(x-y)<1.e-9 for x,y in zip(a,b)), (a,b)


def episode(name,n,offset):
    return dict(clip=name, state=[[offset+0.01*t+j*.001 for j in range(48)]
                                  for t in range(n)],
                eef=[[0.02*t+j*.001 for j in range(9)] for t in range(n)],
                stamp=[t/30 for t in range(n)],
                overlay=[True]*n, tip_mm=[1.5]*n)


def targets(ep):
    q,w=ep["state"],ep["eef"]
    assert len(q)>0 and all(len(v)==48 for v in q)
    assert all(len(v)==9 for v in w)
    return ([q[min(t+1,len(q)-1)][:] for t in range(len(q))],
            [w[min(t+1,len(w)-1)][:] for t in range(len(w))])


def qc(ep,limit_mm=10.,max_dt=.05):
    n=len(ep["state"])
    if not all(len(ep[k])==n for k in ("stamp","overlay","tip_mm")): return False
    if not all(ep["overlay"]): return False
    if not all(math.isfinite(v) and v<=limit_mm for v in ep["tip_mm"]): return False
    if not all(0<b-a<=max_dt for a,b in zip(ep["stamp"],ep["stamp"][1:])): return False
    return all(math.isfinite(v) for row in ep["state"] for v in row)


def main():
    T_wc=[[0,-1,0,1],[1,0,0,2],[0,0,1,3],[0,0,0,1]]
    T_bw=[[1,0,0,-1],[0,1,0,-2],[0,0,1,-3],[0,0,0,1]]
    T_bc=matmul(T_bw,T_wc)
    p=[.2,.4,.6]
    close(xyz(T_bc,p),[-.4,.2,.6])
    close(xyz(inverse(T_bc),xyz(T_bc,p)),p)
    print("PASS SE3 camera-to-world-to-base and inverse:")

    a=episode("clip_A",4,0.)
    b=episode("clip_B",3,2.)
    aq,ae=targets(a)
    bq,be=targets(b)
    close(aq[0],a["state"][1])
    close(aq[-1],a["state"][-1])
    close(ae[1],a["eef"][2])
    close(bq[0],b["state"][1])
    assert aq[-1]!=b["state"][0]
    print("PASS next absolute state and per-episode isolation:")

    for q,goal in zip(a["state"],aq):
        delta=[g-v for g,v in zip(goal,q)]
        close([v+d for v,d in zip(q,delta)],goal)
    assert all(abs(g-v)<1.e-10 for g,v in zip(aq[-1],a["state"][-1]))
    print("PASS absolute-delta roundtrip and terminal repeat:")

    assert qc(a) and qc(b)
    z=episode("invalid_overlay",3,0.)
    z["overlay"][1]=False
    assert not qc(z)
    z=episode("invalid_ik",3,0.)
    z["tip_mm"][0]=100.
    assert not qc(z)
    z=episode("invalid_time",3,0.)
    z["stamp"][1]=z["stamp"][0]
    assert not qc(z)
    print("PASS invalid overlay/IK/time rejection:")

    train={"clip_A","clip_C"}
    test={"clip_B"}
    assert train.isdisjoint(test)
    assert a["clip"] in train and b["clip"] in test
    assert not train.union({"clip_B"}).isdisjoint(test)
    print("PASS source-clip split leakage detector:")
    print("PASS ALL: HuRo CPU contracts")


if __name__=="__main__":
    main()
