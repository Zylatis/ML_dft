from __future__ import division #something about integer division
import numpy as np
from scipy.sparse import diags, bmat
from scipy.sparse.linalg import eigsh, eigs
import matplotlib.pyplot as plt
from operator import add

def make_potential( R1,R2, A1, A2 ):
	potential = []
	for i in range(N):
		for j in range(N):
			r1 = -L/2. + i*dx
			r2 = -L/2. + j*dx
			
			c1r1  = -A1/np.sqrt(1.+(r1-R1)**2)
			c1r2  = -A1/np.sqrt(1.+(r2-R1)**2)
			
			c2r1  = -A2/np.sqrt(1.+(r1-R2)**2)
			c2r2  = -A2/np.sqrt(1.+(r2-R2)**2)
			
			v12 = 1./np.sqrt(1.+(r1-r2)**2)
			
			#~ v1  = r2**2 + r1**2
			#~ v2 = 0
			#~ v12 = 0
			potential.append( c1r1 + c1r2 + c2r1 + c2r2  + v12 )
	return diags( potential, shape = (N**2,N**2) ) 
	
def comp_gs(R1, R2, A1, A2, n ):
	potential = make_potential(R1, R2,A1,A2)
# which='SA', k = n_levels, tol = 0,ncv = 250
	vals, vecs = eigsh(-0.5*H + potential, k = n, which = 'SA', return_eigenvectors = True) #want smallest algebraic not magnitude, need bound state
	vecsE = vecs
	vecs = [ x[n-1] for x in vecs ]
	
	density = [0]*N
	for i in range(N):
		density = list( map(add, density,vecs[N*i:(i+1)*N] ) ) # convert to density by integrating out one part of flat 2D vector

	for i in range(N):
		density[i] = np.abs(density[i])**2
	norm = sum(density)*dx
	density = density/norm
	#~ print density[int(N/2)]
	#~ print vecs[int(N/2)]
	return [ vals, density, vecsE ]
	
	
L = 50
N = 300
dx = L/(N-1.)
x_points = np.linspace(-L/2,L/2,N)

B  = diags([1., -4., 1.], [-1,0, 1], shape=(N, N))
I = diags(1., shape=(N, N))

# tidy up, pinched from SO but general idea same as before
H = bmat([[B if i == j else I if abs(i-j)==1
                else None for i in range(N)]
                for j in range(N)], format='bsr')/dx**2
                
R1 = 5
R2 = -5
A1 = 1
A2 = 1

E, nd, vecsE = comp_gs(R1,R2, A1, A2, 3)
print E
out_nd = np.asarray(nd)
i = 0;
np.savetxt("train_data/" + str(i) + ".dat", out_nd)

V1 = [ -A1/np.sqrt(1.+( -L/2. + i*dx-R1)**2) + -A2/np.sqrt(1.+( -L/2. + i*dx-R2)**2) for i in range(N) ]
plt.plot(x_points, V1)
plt.plot(x_points, nd)

#~ plt.show()
plt.savefig('temp.png')
