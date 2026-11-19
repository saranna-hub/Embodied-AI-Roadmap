#!/usr/bin/env python3
"""Synthetic CPU geometry contracts for EgoForce RSS; no official neural weights."""
import math

def dot(a, b):
    return sum(x*y for x, y in zip(a, b))
def unit(a):
    n=math.sqrt(dot(a,a))
    if n<1e-12: raise ValueError("zero ray")
    return [x/n for x in a]
def solve(A,b):
    x=[list(map(float,A[i]))+[float(b[i])] for i in range(3)]
    for i in range(3):
        j=max(range(i,3),key=lambda k:abs(x[k][i]))
        x[i],x[j]=x[j],x[i]
        if abs(x[i][i])<1e-9: raise ValueError("unobservable ray system")
        p=x[i][i]
        x[i]=[v/p for v in x[i]]
        for j in range(3):
            if i==j: continue
            q=x[j][i]
            x[j]=[u-q*v for u,v in zip(x[j],x[i])]
    return [x[i][3] for i in range(3)]
def rss(J,D,W):
    assert len(J)==len(D)==len(W)
    M=[[0.0]*3 for _ in range(3)]
    m=[0.0]*3
    for j,d,w in zip(J,D,W):
        if w<=0: continue
        d=unit(d)
        P=[[(1.0 if i==k else 0.0)-d[i]*d[k] for k in range(3)] for i in range(3)]
        for i in range(3):
            for k in range(3): M[i][k]+=w*P[i][k]
            m[i]+=w*dot(P[i],j)
    return solve(M,[-u for u in m])
def project(j,t,fx=950,fy=920,cx=640,cy=360):
    p=[j[i]+t[i] for i in range(3)]
    assert p[2]>0
    return (fx*p[0]/p[2]+cx,fy*p[1]/p[2]+cy)
def ray(uv,fx=950,fy=920,cx=640,cy=360):
    return unit([(uv[0]-cx)/fx,(uv[1]-cy)/fy,1])
def gap(x,y):
    return math.sqrt(sum((a-b)**2 for a,b in zip(x,y)))

J=[[0,0,0],[0.12,0.01,0.03],[-0.08,0.04,0.01],[0.04,0.11,-0.02],[-0.10,-0.07,0.06],[0.13,-0.04,-0.01]]
T=[0.13,-0.1,0.82]
uv=[project(j,T) for j in J]
D=[ray(u) for u in uv]
assert gap(rss(J,D,[1]*len(J)),T)<1e-8
print("PASS exact ray closure")
bad=D[:]
bad[0]=ray((100,100))
assert gap(rss(J,bad,[0,1,1,1,1,1]),T)<1e-8
assert gap(rss(J,bad,[1]*len(J)),T)>0.01
print("PASS confidence mask outlier")
try: rss(J,[[0,0,1]]*len(J),[1]*len(J))
except ValueError as e: assert "unobservable" in str(e)
else: raise AssertionError("failed to reject parallel rays")
print("PASS parallel-ray degeneracy")
wrong=[ray(u,fx=1200,fy=1200) for u in uv]
assert gap(rss(J,wrong,[1]*len(J)),T)>0.01
print("PASS bad intrinsics negative control")
ang=math.pi/4;c=math.cos(ang);s=math.sin(ang)
R=[[c,-s,0],[s,c,0],[0,0,1]]
tw=[1.3,-0.2,0.4];pc=[0.2,0.1,0.9]
pw=[dot(row,pc)+p for row,p in zip(R,tw)]
inv=[sum(R[k][i]*(pw[k]-tw[k]) for k in range(3)) for i in range(3)]
assert gap(inv,pc)<1e-9
print("PASS camera world SE3 inverse")
print("PASS ALL: EgoForce RSS CPU geometry")
