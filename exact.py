from __future__ import division #something about integer division
import numpy as np
from scipy.sparse import diags, bmat
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

L = 1.
N = 200
dx = L/(N-1.)
#~ x = np.linspace(0,L,N)
#~ H = diags([1., -2., 1.], [-1,0, 1], shape=(N, N))/dx**2

#~ potential = [ -1./np.sqrt((dx*(r-N/2))**2 + 0.1) for r in range(0, N) ]
#~ V = diags(potential, 0,  shape=(N, N))

#~ n_levels = 1
#~ vals, vecs = eigsh(-0.5*H , which='SA', k = n_levels) #want smallest algebraic not magnitude, need bound state
#~ print vals
#~ exit(0)
#~ print vals
#~ plt.plot(x,vecs[:,0])
#~ plt.plot(x,vecs[:,1])

#~ plt.plot(x,potential)
#~ plt.show()


B  = diags([1., -4., 1.], [-1,0, 1], shape=(N, N))
I = diags(1., shape=(N, N))

# tidy up, pinched from SO but general idea same as before
H = bmat([[B if i == j else I if abs(i-j)==1
                else None for i in range(N)]
                for j in range(N)], format='bsr')/dx**2

potential = []
for i in range(N):
	for j in range(N):
		r1 = -L/2. + i*dx
		r2 = -L/2. + j*dx
		v1  = -2./np.sqrt(1.+r1**2)
		v2  = -2./np.sqrt(1.+r2**2)
		v12 = 1./np.sqrt(1.+(r1-r2)**2)
		
		#~ v1  = r2**2 + r1**2
		#~ v2 = 0
		#~ v12 = 0
		potential.append( v1 + v2 + v12 )
		
potential = diags( potential, shape = (N**2,N**2) ) 

n_levels = 8
vals, vecs = eigsh(-0.5*H + potential, which='SA', k = n_levels, tol = 0) #want smallest algebraic not magnitude, need bound state
print vals
#~ print vals[3]-vals[2]
#~ print vals[2]-vals[1]
#~ print vals[1]-vals[0]
