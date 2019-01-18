import numpy as np
import random
from scipy.optimize import minimize, curve_fit

# un-normalised gaussians
def gaussian( x, pars ):
	A = pars[0]
	centre = pars[1]
	sigma = pars[2]
	return A*np.exp( -((x-centre)**2)/(sigma**2)) 
	

# parameters come in sets of 3: weight, centre, width
def model( X, *argv ):
	y = np.zeros(len(X))
	par_set = []
	temp = []
	for arg in argv:
		temp.append(arg)
		if(len(temp)==3):
			par_set = par_set + [temp]
			temp = []

	for par in par_set:
		contrib = np.array(gaussian(X, par))
		y = np.add(y, contrib)
	return y + 1.


# bounds seems to mess with things here, possibly we need a custom solver i.e. simulated annealing
def gaussian_exp( x , y, basis_size, Arange = [0.05,5],crange = [-np.inf,np.inf],sigmarange = [0.01,10.] ):
	
	lower_bound = sum( [ [ Arange[0],crange[0],sigmarange[0]] for n in range(basis_size)], [])
	upper_bound = sum( [ [ Arange[1],crange[1],sigmarange[1]] for n in range(basis_size)], [])
	p0 = sum( [ [ random.uniform(1.,5.) , random.uniform(-15.,15.), random.uniform(0.5,5) ] for n in range(basis_size)], [])
	popt,pcov = curve_fit( model, x , y, p0 = p0, method = 'trf',  max_nfev = 50000)
	return popt
	
#~ gtol = 10**(-15), ftol = 10**(-15), xtol = 10**(-15)
