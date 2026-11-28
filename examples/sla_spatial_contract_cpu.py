"""Independent synthetic SLA interface tests, not official model inference."""
import math


def close(a,b):
    assert math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-9),(a,b)


def sigmoid(x):
    return 1/(1+math.exp(-x))


def masked_mean(losses,masks):
    return sum(sum(x*m for x,m in zip(l,mask))/max(1,sum(mask))
               for l,mask in zip(losses,masks))/len(losses)


def bilinear(grid,y,x):
    i,j=math.floor(y),math.floor(x)
    a,b=y-i,x-j
    return (1-a)*((1-b)*grid[i][j]+b*grid[i][j+1])+a*((1-b)*grid[i+1][j]+b*grid[i+1][j+1])


def mm(a,b):
    return [[sum(x*y for x,y in zip(row,col)) for col in zip(*b)] for row in a]


def tr(a):
    return list(zip(*a))


def rz(t):
    c,s=math.cos(t),math.sin(t)
    return [[c,-s,0],[s,c,0],[0,0,1]]


def rx(t):
    c,s=math.cos(t),math.sin(t)
    return [[1,0,0],[0,c,-s],[0,s,c]]


def main():
    close(sum(sigmoid(0)*q for q in (2,4,6)),6)
    assert sum(sigmoid(0) for _ in range(3))>1
    print("PASS sigmoid is not normalized across layers")
    close(masked_mean([[1,9,9],[0,0,0]],[[1,0,0],[1,1,1]]),.5)
    close(masked_mean([[3,4]],[[0,0]]),0)
    print("PASS samplewise mask denominator and empty mask")
    close(bilinear([[0,2],[4,6]],.5,.5),3)
    close(bilinear([[100,102],[104,106]],.5,.5),103)
    print("PASS per-view bilinear interpolation isolation")
    a0=rx(.6); a1=rz(.9)
    delta=mm(a1,tr(a0))
    correct=mm(delta,a0)
    wrong=mm(a0,delta)
    for x,y in zip(sum(correct,[]),sum(a1,[])):
        close(x,y)
    assert max(abs(x-y) for x,y in zip(sum(wrong,[]),sum(a1,[])))>.1
    rel=mm(rz(math.radians(-179)),tr(rz(math.radians(179))))
    close(math.degrees(math.atan2(rel[1][0],rel[0][0])),2)
    assert [p-1 for p in (2,4,7)]==[1,3,6]
    print("PASS base-frame rotation composition and chunk anchor")
    K=[[400,0,320],[0,420,240],[0,0,1]]
    transformed=mm([[.5,0,-10],[0,.25,-5],[0,0,1]],K)
    close(transformed[0][0],200);close(transformed[0][2],150)
    close(transformed[1][1],105);close(transformed[1][2],55)
    print("PASS camera crop and resize intrinsics")
    print("PASS ALL: SLA independent synthetic spatial contracts")


if __name__=="__main__":
    main()
