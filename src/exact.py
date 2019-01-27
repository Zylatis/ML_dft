from scipy.sparse import diags, bmat
from scipy.sparse.linalg import eigsh, eigs

import matplotlib.pyplot as plt
from operator import add
import numpy as np
import random
from modules import basis
import time

# Fn to make potential with 2 nuclei + e-e term
# (clamped nuclei)
# R1, R2 = position (relative to r = 0) of nuclei
# A1 and A2 are atomic numbers (charge) of each nuclei

def pot2(i,j ):
	R1 = 1
	R2 = 2
	A1 = 1
	A2 = 2
	# Define position
	r1 = -L/2. + i*dx
	r2 = -L/2. + j*dx
	
	# e1 interaction with nuclei
	c1r1  = -A1/np.sqrt( 1.+(r1-R1)**2 )
	c1r2  = -A1/np.sqrt( 1.+(r2-R1)**2 )

	# e2 interaction with nuclei			
	c2r1  = -A2/np.sqrt( 1.+(r1-R2)**2 )
	c2r2  = -A2/np.sqrt( 1.+(r2-R2)**2 )
	
	# e-e potential
	v12 = 1./np.sqrt( 1.+(r1-r2)**2 )
	
	return c1r1 + c1r2 + c2r1 + c2r2 + v12 

def make_potential( R1, R2, A1, A2 ):
	potential = []
	t0 = time.time()
	points = np.asarray([ x for x in range(N)])
	xgrid,ygrid = np.meshgrid(points,points)
	
	pot3 = pot2(xgrid,ygrid)
	t1 = time.time()
	print t1-t0
	# List to store values (flattened 2D list)
	# potential = []
	t0 = time.time()
	potential = [ pot2(i,j) for i in range(N) for j in range(N)]
	t1 = time.time()
	print t1-t0
	print("-")
	print potential-pot3.flatten()
	print 
	# # Loop over both electronic coordinates
	# for i in range(N):
	# 	for j in range(N):
			
	# 		# Define position
	# 		r1 = -L/2. + i*dx
	# 		r2 = -L/2. + j*dx
			
	# 		# e1 interaction with nuclei
	# 		c1r1  = -A1/np.sqrt( 1.+(r1-R1)**2 )
	# 		c1r2  = -A1/np.sqrt( 1.+(r2-R1)**2 )

	# 		# e2 interaction with nuclei			
	# 		c2r1  = -A2/np.sqrt( 1.+(r1-R2)**2 )
	# 		c2r2  = -A2/np.sqrt( 1.+(r2-R2)**2 )
			
	# 		# e-e potential
	# 		v12 = 1./np.sqrt( 1.+(r1-r2)**2 )
			
	# 		# Add sum to list
	# 		potential.append( c1r1 + c1r2 + c2r1 + c2r2 + v12 )
			
	# Return potential as sparse matrix		
	exit(0)

	return diags( potential, shape = (N**2,N**2) ) 
	
# Fn to compute ground state energy and density 
# (can in principle compute any levels, hence the 'n'
# so perhaps poorly named)
def comp_gs( R1, R2, A1, A2, n ):
	
	# Get potential
	potential = make_potential( R1, R2, A1, A2 )
	


	# Solve eigensystem, ensuring we return the lowest algebraic values first
	# as we will have negative values (bound states)

	vals, vecs = eigsh(-0.5*H + potential, k = n, which = 'SA') #want smallest algebraic not magnitude, need bound state

	# Take the appropriate level ( could in principle return them all + densities, will refactor later)
	vecs = [ x[n-1] for x in vecs ]
	
	# Make empty list to store 1D density (we want just 1-body density so integrate out all but 1
	# of the electronic coordinates)
	# We start at zero because we are doing an integral so we do a sum in the loop
	density = [0]*N
	for i in range(N):
		# This first takes the 2D eigenvector selected above and integrates out the 2nd coordinate
		# by adding subsequent 'blocks' of size N. As these are identical particles the particular
		# flattening format doesn't matter
		density = list( map(add, density,vecs[N*i:(i+1)*N] ) ) 

	# Really what we have above is the sum of the wfn but we need the abs squared.
	# Actually, what we should be doing is taking th sum of the magnitude not the magnitude of the sum, however
	# We are dealing with wfns which are purely real so it doesn't matter
	for i in range(N):
		density[i] = np.abs( density[i] )**2
	
	# Calculate norm and normalise appropriately.
	# This is just for plotting, we are not computing observables (we already have the energy)
	norm = sum( density )*dx
	density = density/norm

	# Return list of *all* energy eigenvalues up to n
	# as well as the density of the n-th state
	return [ vals, density ]
	

################################################################################################
# MAIN PROGRAM
################################################################################################

n_round = 2
# COMPUTATIONAL PRELIMS
L = 70 	# box size
N = 150 # number of points
dx = L/(N-1.) # grid spacing
x_points = np.linspace(-L/2,L/2,N) # spatial grid used for plotting
# Block component for 2D Laplacian matrix
B  = diags([1., -4., 1.], [-1,0, 1], shape=(N, N))
# Identity matrix (could be more pythonic here0
I = diags([1.], shape=(N, N))

# Construct 2D Laplacian matrix from above blocks.
# This is the main part which will need to be changed if we go to 3 electrons
H = bmat([[B if i == j else I if abs(i-j)==1
                else None for i in range(N)]
                for j in range(N)], format='csr')/dx**2


# SYSTEM LAYOUT   
# Location and charge of nuclei             
random.seed(4)
n_samples = 3
Rmin = -10.
Rmax = 10.
Amin = 0.1
Amax = 3.
energies = []
densities = []
for k in range(n_samples):
	R1 = random.uniform(Rmin,Rmax)
	R2 = random.uniform(Rmin,Rmax)
	A1 = random.uniform(Amin,Amax)
	A2 = random.uniform(Amin,Amax)
	E, nd  = comp_gs(R1,R2, A1, A2, 1 )
	energies.append( E )
	densities.append ( nd )
	# Prepare density export and plotting
	
	if k%100 == 0:
		print( k, round(E[0],n_round))#,round(R1,n_round), round(R2,n_round)
	
energies = np.asarray(energies)
#~ densities = np.asarray( densities )
np.savetxt( "../train_data/densities.dat", densities )
#~ np.savetxt( "train_data/energies.dat", energies )


i = 0
x = np.linspace(-L/2,L/2, N)
xx = np.linspace(0,N,N)
for soln in densities:
	bf_pars = basis.gaussian_exp( xx, soln , 5 )
	plt.plot( soln )
	plt.plot( basis.model( xx,*bf_pars ))

	plt.savefig(str(i) + ".png" )
	plt.clf()
	i = i+1
# print(bf_pars)

# plt.plot( densities[n_samples-1] )
# plt.plot( basis.model( xx,*bf_pars ))
# plt.show()
# End of file
