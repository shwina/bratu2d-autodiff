import sys, petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
import numpy as np

# this user class is an application
# context for the nonlinear problem
# at hand; it contains some parametes
# and knows how to compute residuals

class Bratu2D:

    def __init__(self, nx, ny, alpha, impl='python'):
        self.nx = nx # x grid size
        self.ny = ny # y grid size
        self.alpha = alpha
        if impl == 'python':
            from bratu2dnpy import bratu2d, bratu2d_jac
            order = 'c'
        elif impl == 'theano':
            from bratu2dtheano import bratu2d, bratu2d_jac
            order = 'c'
        else:
            raise ValueError('invalid implementation')
        self.compute = bratu2d
        self.compute_jacobian = bratu2d_jac
        self.order = order

    def evalFunction(self, snes, X, F):
        nx, ny = self.nx, self.ny
        alpha = self.alpha
        order = self.order
        x = X.getArray(readonly=1).reshape(nx, ny, order=order)
        f = F.getArray(readonly=0).reshape(nx, ny, order=order)
        self.compute(alpha, x, f)

    def evalJacobian(self, snes, X, J, P):
        nx, ny = self.nx, self.ny
        alpha = self.alpha
        order = self.order
        x = X.getArray(readonly=1).reshape(nx, ny, order=order)
        self.compute_jacobian(alpha, x, J)
        
# convenience access to
# PETSc options database
OptDB = PETSc.Options()

nx = OptDB.getInt('nx', 32)
ny = OptDB.getInt('ny', nx)
alpha = OptDB.getReal('alpha', 6.8)
impl  = OptDB.getString('impl', 'theano')

# create application context
# and PETSc nonlinear solver
appc = Bratu2D(nx, ny, alpha, impl)
snes = PETSc.SNES().create()

# register the function in charge of
# computing the nonlinear residual
f = PETSc.Vec().createSeq(nx*ny)
snes.setFunction(appc.evalFunction, f)

# register the function in charge of
# computing the jacobian
Jmat = PETSc.Mat().create()
Jmat.setSizes((nx*ny, nx*ny))
Jmat.setType('dense')
Jmat.setUp()
snes.setJacobian(appc.evalJacobian, Jmat, Jmat)

# configure the nonlinear solver
# to use a matrix-free Jacobian
#snes.setUseMF(True)
snes.getKSP().setType('cg')
snes.setFromOptions()

# solve the nonlinear problem
b, x = None, f.duplicate()
x.set(0) # zero inital guess
snes.solve(b, x)

# save solution to output file
outfile = OptDB.getString('outfile', 'z.txt')
from numpy import mgrid
X, Y =  mgrid[0:1:1j*nx,0:1:1j*ny]
Z = x[...].reshape(nx,ny)
np.savetxt(outfile, Z)

if OptDB.getBool('plot', True):
    da = PETSc.DMDA().create([nx,ny])
    u = da.createGlobalVec()
    x.copy(u)
    draw = PETSc.Viewer.DRAW()
    OptDB['draw_pause'] = 1
    draw(u)

if OptDB.getBool('plot_mpl', False):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as pylab
    except ImportError:
        PETSc.Sys.Print("matplotlib not available")
    else:
        pylab.figure()
        pylab.contourf(X,Y,Z)
        pylab.colorbar()
        pylab.plot(X.ravel(),Y.ravel(),'.k')
        pylab.axis('equal')
        pylab.savefig('result.png')
