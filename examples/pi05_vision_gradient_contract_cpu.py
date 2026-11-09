#!/usr/bin/env python3
"""CPU-only math/data contracts: research design, NOT official openpi/GPU reproduction."""
import math


def dot(a, b):
    assert len(a) == len(b) and len(a) > 0
    return sum(x * y for x, y in zip(a, b))


def norm(a):
    return math.sqrt(dot(a, a))


def cosine_shared(g_flow, g_teacher):
    f, t = norm(g_flow), norm(g_teacher)
    if f == 0 or t == 0:
        return None  # Undefined, not 0: frozen/unconnected params must be excluded
    return dot(g_flow, g_teacher) / (f * t)


def protect_flow(g_flow, g_teacher, eps=1e-12):
    """Asymmetric research variant: project ONLY teacher gradient if conflicting."""
    assert len(g_flow) == len(g_teacher)
    if dot(g_flow, g_teacher) >= 0 or norm(g_flow) == 0:
        return list(g_teacher)
    scale = dot(g_flow, g_teacher) / (dot(g_flow, g_flow) + eps)
    return [t - scale * f for f, t in zip(g_flow, g_teacher)]


def weighted_teacher_loss(patch_sq_errors, camera_present, patch_visible):
    assert len(patch_sq_errors) == len(camera_present) == len(patch_visible)
    valid = [err for err, cam, vis in zip(patch_sq_errors, camera_present, patch_visible)
             if cam and vis]
    return sum(valid) / len(valid) if valid else 0.0


def training_inputs(rgb_current, teacher_current, action_chunk):
    assert rgb_current is not None and teacher_current is not None and action_chunk is not None
    return {'rgb_current': rgb_current, 'teacher_target': teacher_current, 'action_target': action_chunk}


def inference_inputs(rgb_current):
    # Teacher and action targets are deliberately absent from inference signature and dict.
    return {'rgb_current': rgb_current}


def euler_reverse(noise, action, steps):
    # openpi convention: x(t)=t*noise+(1-t)*action; v=noise-action; t decreases 1 to 0.
    assert steps > 0
    x = noise
    v = noise - action
    for _ in range(steps):
        x -= v / steps
    return x


def run():
    gf, gt = [1.0, 0.0], [-1.0, 2.0]
    assert cosine_shared(gf, gt) < 0
    pt = protect_flow(gf, gt)
    assert all(abs(a-b)<1e-10 for a,b in zip(pt, [0.0,2.0]))
    assert dot(gf, pt) > -1e-10
    print('PASS negative-angle projection: (-1,2) -> (0,2)')

    orth = [0.0, 4.0]
    assert cosine_shared(gf, orth) == 0.0
    assert protect_flow(gf, orth) == orth
    assert protect_flow(gf, [2.0, 4.0]) == [2.0,4.0]
    assert cosine_shared([0.0,0.0], orth) is None
    print('PASS orthogonal / positive / zero-gradient: no false conflict')

    # First-order directional derivative of Flow loss along negative joint update.
    lam = 0.4
    joint = [f + lam*t for f,t in zip(gf, pt)]
    assert dot(gf, joint) > 0.0
    assert dot(gf, [f + lam*t for f,t in zip(gf, gt)]) > 0.0
    print('PASS first-order action-loss protection on shared parameters')

    # Absent camera with giant error must not contaminate feature loss.
    observed = weighted_teacher_loss([0.04, 0.09, 10_000.0],
                                     [True,True,False],[True,True,True])
    assert abs(observed-0.065)<1e-12
    assert weighted_teacher_loss([99.], [False],[True]) == 0
    print('PASS valid-camera and patch masks: absent camera excluded')

    tr = training_inputs('rgb_t','dino_teacher_t', 'action_t:t+H')
    inf = inference_inputs('rgb_t')
    assert 'teacher_target' in tr and 'action_target' in tr
    assert set(inf.keys()) == {'rgb_current'}
    print('PASS train/inference causality: teacher and future actions absent online')

    for n in (1, 4, 10, 25):
        assert abs(euler_reverse(0.75, -0.13, n) - (-0.13)) < 1e-10
    print('PASS reverse-time Flow Euler: t=1 noise -> t=0 action')
    print('PASS ALL: pi05 visual distillation gradient geometry CPU contracts')


if __name__=='__main__':
    run()
