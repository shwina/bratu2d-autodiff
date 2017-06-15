# fil: bratu2dnpy.py
from numpy import exp
import numpy as np

def bratu2d(alpha, x, f):
    # get 'exp' from numpy
    # setup 5-points stencil
    u  = x[1:-1, 1:-1] # center
    uN = x[1:-1,  :-2] # north
    uS = x[1:-1, 2:  ] # south
    uW = x[ :-2, 1:-1] # west
    uE = x[2:,   1:-1] # east
    # compute nonlinear function
    nx, ny = x.shape
    dx = 1.0/(nx-1) # x grid spacing
    f[:,:] = x
    f[1:-1, 1:-1] = \
         (2*u - uE - uW)  \
       + (2*u - uN - uS)  \
       - alpha * exp(u)  * (dx*dx)

def bratu2d_jac(alpha, x, M):
    nx, ny = x.shape
    dx, dy = 1./(nx-1), 1./(ny-1)
    for i in range(nx):
        for j in range(ny):
            row = j + ny*i
            if (i == 0 or j == 0 or i == (nx-1) or j == (ny-1)):
                col = j + ny*i
                M.setValue(row, col, 1.0)
            else:
                col = (j-1) + ny*i
                #M.setValue(row, col, -1./(dx*dx))
                M.setValue(row, col, -1.)
                col = (j+1) + ny*i
                #M.setValue(row, col, -1./(dx*dx))
                M.setValue(row, col, -1)
                col = j + ny*(i-1)
                #M.setValue(row, col, -1./(dx*dx))
                M.setValue(row, col, -1.)
                col = j + ny*(i+1)
                #M.setValue(row, col, -1./(dx*dx))
                M.setValue(row, col, -1.)
                col = j + ny*i
                #M.setValue(row, col, 4./(dx*dx) - alpha * exp(x[i, j]))
                M.setValue(row, col, 4. - alpha * exp(x[i, j]) * dx*dx)
                #M.setValue(row, col, np.random.rand())
    M.assemble()
