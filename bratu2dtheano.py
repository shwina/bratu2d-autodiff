import theano
import theano.tensor as T
import numpy as np

def bratu2dfunc(alpha, x):
    f = x.copy()
    u  = x[1:-1, 1:-1] # center
    uN = x[1:-1,  :-2] # north
    uS = x[1:-1, 2:  ] # south
    uW = x[:-2,  1:-1] # west
    uE = x[2:,   1:-1] # east
    # compute nonlinear function
    nx, ny = x.shape
    hx = 1.0/(nx-1) # x grid spacing
    hy = 1.0/(ny-1) # y grid spacing
    return T.set_subtensor(f[1:-1, 1:-1], (2*u - uE - uW) * (hy/hx) \
        + (2*u - uN - uS) * (hx/hy) \
        - alpha * T.exp(u)  * (hx*hy))

alpha = T.scalar('alpha')
x = T.matrix('x')
func = theano.function(
        [alpha, x], bratu2dfunc(alpha, x))
jac = theano.function(
        [alpha, x],
        theano.gradient.jacobian(bratu2dfunc(alpha, x).flatten(), x))

def bratu2d(alpha, x, f):
    f[:,:] = func(alpha,x)

def bratu2d_jac(alpha, x, M):
    nx, ny = x.shape
    J = jac(alpha, x)
    J = J.reshape((nx*ny), (nx*ny))
    M_ary = M.getDenseArray()
    M_ary[...] = J
