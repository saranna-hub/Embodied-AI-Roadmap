#!/usr/bin/env python3
"""EgoActionBench formula and time/hand contracts: standard library only.

Independent synthetic illustration, NOT the authors' unpublished evaluator.
Fixed hand/content scores emulate a text judge; they do not implement Qwen3-32B.
"""
from dataclasses import dataclass
from math import isclose

@dataclass(frozen=True)
class A:
    start: float
    end: float
    hand: str
    caption: str = ""

def overlap(a, b):
    return max(0.0, min(a.end, b.end) - max(a.start, b.start))

def valid(a):
    return 0 <= a.start < a.end and a.hand in ("L", "R", "B")

def omin(a, b):
    assert valid(a) and valid(b)
    return overlap(a, b) / min(a.end - a.start, b.end - b.start)

def tiou(a, b):
    assert valid(a) and valid(b)
    return overlap(a, b) / (a.end - a.start + b.end - b.start - overlap(a,b))

def compatible(g, p):
    return (p.hand == "B") if g.hand == "B" else p.hand in ("L", "R")

def candidates(g, preds, tau=.5):
    return [p for p in preds if valid(p) and compatible(g,p) and omin(g,p)>=tau]

g=A(2,3,"L")
p=A(0,10,"L")
assert isclose(omin(g,p),1.0) and isclose(tiou(g,p),.1)
assert p in candidates(g,[p])
assert isclose(overlap(A(0,1,"L"), A(1,2,"L")),0)
print("PASS Omin-vs-tIoU long-interval negative control; half-open boundary")

assert A(2,3,"R") in candidates(g,[A(2,3,"R")])
assert not candidates(A(2,3,"B"),[A(2,3,"L"),A(2,3,"R")])
assert A(2,3,"B") in candidates(A(2,3,"B"),[A(2,3,"B")])
print("PASS Left-Right candidate, Both isolation")

# The semantic score includes hand separately; each REQUIRED field has equal weight
field={"verb":1.0, "objects":1.0, "motion":0.0}
content=sum(field.values())/len(field)
semantic=(0+sum(field.values()))/(1+len(field)) # wrong hand
assert isclose(content,2/3) and isclose(semantic,.5)
assert isclose((1+sum(field.values()))/(1+len(field)),.75)
print("PASS required fields and acting-hand semantic contribution")

# One wide caption covers two reference intervals but 1:1 maximum-cardinality
gold=[A(0,1,"L"),A(1,2,"L")]
preds=[A(0,2,"L")]
edges={i:[j for j,p in enumerate(preds) if p in candidates(gold[i],preds)] for i in range(len(gold))}
def max_match(edges):
    """Small DFS augmenting-path maximum bipartite cardinality matching."""
    owner={}
    def augment(left,seen):
        for right in edges[left]:
            if right in seen: continue
            seen.add(right)
            if right not in owner or augment(owner[right],seen):
                owner[right]=left
                return True
        return False
    return sum(augment(i,set()) for i in sorted(edges))
tp=max_match(edges)
f1=100*2*tp/(len(gold)+len(preds))
assert tp==1 and isclose(f1,200/3)
assert [tiou(x,preds[0]) for x in gold]==[.5,.5]
print("PASS one prediction cannot be two F1 true positives")

# Repeated same-action predictions all remain in denominator, including invalid interval
gold=[A(0,1,"R")]
preds=[A(0,1,"R"),A(0,1,"R"),A(3,2,"R")]
edges={0:[i for i,p in enumerate(preds) if valid(p) and p in candidates(gold[0],preds)]}
tp=max_match(edges)
assert tp==1 and len(preds)==3 and isclose(100*2*tp/(1+3),50)
print("PASS duplicates and invalid intervals count toward FP")

# Same content one second timestamp in frame units must be converted by original fps
fps=50
start_frame,end_frame=100,115
assert isclose(start_frame/fps,2.0) and isclose(end_frame/fps,2.3)
assert not isclose((end_frame-start_frame)/fps, 15)
print("PASS frame-to-second explicit timebase; short-action 0.3s")

# Consistent 0-100 reporting and reward 0-1: no mixing.
S,I,F=.8,.6,.4
reward=.5*S+.25*I+.25*F
semantic=80.0; mt_iou=60.0; f1=40.0
temporal=(mt_iou+f1)/2
score=(semantic+temporal)/2
assert isclose(reward,.65) and isclose(score,65)
print("PASS SFT-GRPO reward scale and EgoActionScore percentages")

# Sampling illustration: duration .28s at 2fps may fit between samples
action=(.11,.39)
sample_times=[0,.5,1.0]
assert not any(action[0]<=t<action[1] for t in sample_times)
assert any(action[0]<=t<action[1] for t in [i/10 for i in range(11)])
print("PASS short-action sparse-fps missed-frame negative control")

print("PASS ALL: EgoActionBench synthetic temporal/semantic contracts")
