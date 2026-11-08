"""Educational FlowDPG numeric contracts (stdlib only; no author weights)."""
from math import sqrt, isclose

def sub(a, b):
    return [x-y for x,y in zip(a,b)]

def add(a, b):
    return [x+y for x,y in zip(a,b)]

def mul(k, a):
    return [k*x for x in a]

def norm(a):
    return sqrt(sum(x*x for x in a))

def check_near(a,b,tol=1e-10):
    assert norm(sub(a,b)) < tol, (a,b)

def expectile(q,v,tau):
    d=q-v
    return (tau if d>=0 else 1-tau)*d*d

def sarm_reward(p0,p1,z0,z1,beta,terminal):
    return (p1-p0) + beta*int(z1>z0) + terminal

def flowdpg_target(x0,x1,t,v,g,alpha,eps=1e-8):
    xt=add(mul(1-t,x0),mul(t,x1))
    u=sub(x1,x0)
    a_hat=add(xt,mul(1-t,v))
    delta=mul(alpha*norm(u)/(norm(g)+eps), g)
    a_star=add(a_hat,delta)
    return xt,u,a_hat,delta,sub(a_star,x0)

def main():
    x0=[-1.2,0.4,0.7,0.1]; x1=[0.8,-0.2,1.1,0.9]
    u=sub(x1,x0)
    for t in [0.01,0.20,0.65,0.99]:
        xt,_,ahat,shift,target=flowdpg_target(x0,x1,t,u,[0.3,-0.2,0.1,0.4],0.05)
        check_near(ahat,x1)
        check_near(target,add(u,shift))
        badv=add(u,[0.2,-0.1,0.04,0.05])
        _,_,bad_hat,_,_=flowdpg_target(x0,x1,t,badv,[0.3,-0.2,0.1,0.4],0.05)
        assert isclose(norm(sub(bad_hat,x1))**2,(1-t)**2*norm(sub(badv,u))**2,rel_tol=1e-8)
    print('PASS projection and consistency identity: all t')
    goal=[1.0,0.0,1.5,1.0]
    g=sub(goal,x1)
    _,_,ahat,shift,target=flowdpg_target(x0,x1,0.4,u,g,0.05)
    assert sum(a*b for a,b in zip(g,shift))>0
    assert norm(sub(add(ahat,shift),goal)) < norm(sub(ahat,goal))
    assert abs(norm(shift)-0.05*norm(u)) < 1e-6
    assert all(isclose(x,y,abs_tol=1e-10) for x,y in zip(target,add(u,shift)))
    print('PASS critic gradient improves local quadratic Q; scaled norm and target')
    _,_,_,shift0,_=flowdpg_target(x0,x1,0.4,u,[0,0,0,0],0.05)
    check_near(shift0,[0,0,0,0])
    assert all(isclose(expectile(0,v,.8),.2*v*v) for v in [1,2])
    assert isclose(expectile(2,0,.8),3.2)
    print('PASS zero gradient safety and upper expectile asymmetry')
    assert isclose(sarm_reward(.3,.6,2,3,.2,0),.5)
    assert isclose(sarm_reward(.6,.3,3,3,.2,0),-.3)
    assert isclose(sarm_reward(.4,.4,4,4,.2,1),1)
    print('PASS progress + transition + terminal reward; backward-progress penalty')
    noisy_actor=[u[0]+.4,u[1],u[2],u[3]]
    _,_,_,_,ut_star=flowdpg_target(x0,x1,.5,u,g,.05)
    def weighted_loss(pred,lam):
        return lam*norm(sub(pred,ut_star))**2+(1-lam)*norm(sub(pred,u))**2
    assert isclose(weighted_loss(noisy_actor,0),norm(sub(noisy_actor,u))**2)
    assert isclose(weighted_loss(noisy_actor,1),norm(sub(noisy_actor,ut_star))**2)
    print('PASS BC anchor and distillation interpolation')
    print('PASS ALL: FlowDPG CPU mathematical contracts (not official reproduction)')

if __name__=='__main__':
    main()
