#!/usr/bin/env python3
"""CPU-only synthetic mathematical/ABI contracts for tau0-VLA tutorial.
NOT official TTC implementation; requires Python standard library only."""
import math

def choose_ttc(probs_all, probs_memory, threshold_all=.65, threshold_mem=.5):
    assert probs_all and probs_memory
    assert all(0 <= p <= 1 for p in probs_all + probs_memory)
    ua = sum(probs_all) / len(probs_all)
    um = sum(probs_memory) / len(probs_memory)
    return ua <= threshold_all or um <= threshold_mem

assert choose_ttc([.9, .8, .6], [.1, .2])
assert not choose_ttc([.9, .8, .8], [.7, .8])
assert choose_ttc([.3, .4], [.9])  # memory alone is not enough
print("PASS confidence router all/memory OR and equality")

# A greedy root ranking is not equivalent to globally-scored depth-2 beam search.
root = [("place_immediately", .8), ("inspect_grip", .6)]
assert max(root, key=lambda x:x[1])[0] == "place_immediately"
children = {"place_immediately": [("slip", -.8)], "inspect_grip": [("stable_place", .7)]}
ranked = sorted([(r + "/" + action, score + bonus)
                 for r, score in root for action, bonus in children[r]], key=lambda x:-x[1])
assert ranked[0][0] == "inspect_grip/stable_place"
assert math.isclose(ranked[0][1], 1.3)
assert 3 + (2-1)*2*3 == 9  # N=3, B=2, D=2
print("PASS beam cumulative score and call budget")

real_memory = {"stage":"grip_unconfirmed", "version":7}
candidate_a = {**real_memory, "stage":"imagined_success"}
candidate_b = {**real_memory, "stage":"imagined_failure"}
candidate_a["version"] = 99
assert real_memory == {"stage":"grip_unconfirmed", "version":7}
assert candidate_b["stage"] == "imagined_failure"
print("PASS branch-local imaginary memory never commits")

# 40D canonical layout; LIBERO only uses EEF xyz/Rot6D and left gripper.
slots={"left_eef": slice(0,9), "right_eef": slice(9,18), "left_grip":18,
       "right_grip":19,"waist":slice(20,22),"base":slice(22,24),
       "left_joint":slice(24,32),"right_joint":slice(32,40)}
active=set(range(9)) | {18}
assert len(active)==10 and max(active)<40
state=[0.0]*40
state[18]=.25
noise=[(i+1)*.07 for i in range(40)]
masked_noise=[v if i in active else 0.0 for i,v in enumerate(noise)]
assert all(masked_noise[i]==0 for i in range(40) if i not in active)
assert set(range(40)) == set().union(set(range(9)),set(range(9,18)),{18,19},
                                     set(range(20,22)),set(range(22,24)),
                                     set(range(24,32)),set(range(32,40)))
print("PASS 40D slot map and LIBERO inactive mask")

# Reverse-time flow: x_tau=tau*epsilon+(1-tau)*a, dx/dtau=epsilon-a.
a, eps = 2.0, 5.0
x=eps
for _ in range(10):
    x += -.1*(eps-a)
assert math.isclose(x,a,abs_tol=1e-12)
bad=eps
for _ in range(10):
    bad += -.1*(a-eps)
assert math.isclose(bad,8.0) and not math.isclose(bad,a)
print("PASS reverse-time flow Euler sign and negative control")

# masked loss: inactive junk must never affect denominator or numerator
target=[0.1]*40
prediction=[0.3 if i in active else 1e6 for i in range(40)]
loss=sum((prediction[i]-target[i])**2 for i in active)/len(active)
assert math.isclose(loss,.04)
all_dim_loss=sum((prediction[i]-target[i])**2 for i in range(40))/40
assert all_dim_loss>1e10
print("PASS masked FM denominator excludes 30 inactive channels")

# Memory update: invalid/imaginary response must NOT erase verified memory
def next_memory(previous, predicted, format_valid):
    return previous if not format_valid else (predicted.strip() or "(empty)")
assert next_memory("grip unconfirmed", "imagined success", False)=="grip unconfirmed"
assert next_memory("grip unconfirmed", "", True)=="(empty)"
print("PASS invalid parse preserves real progress")

# Explicit mismatch between data ABI: HiFi 20D absolute next-state, tau 40D relative
def to_delta(current, next_absolute):
    return next_absolute-current
assert math.isclose(to_delta(.9,1.2),.3)
assert not math.isclose(1.2,.3)
print("PASS absolute-next-state requires relative conversion")

print("PASS ALL: tau0 TTC synthetic contracts")
