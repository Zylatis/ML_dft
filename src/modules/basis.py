import numpy as np
import random
from scipy.optimize import minimize, curve_fit
from scipy.signal import find_peaks
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
	return y 


# bounds seems to mess with things here, possibly we need a custom solver i.e. simulated annealing
def gaussian_exp( x , y, basis_size, Arange = [0.1,20],crange = [0,250],sigmarange = [0.1,300.] ):
	peaks, peak_heights = find_peaks(y, height = 0.1)
	p0_0 = sum( [ [ 0.1 , peaks[n], 10 ] for n in range(len(peaks))], [])
	print(peaks)
	if len(peaks)<basis_size:
		p0_1 = sum( [ [0.1 , np.mean(peaks), 1] for n in range(basis_size - len(peaks))], [])
	p0 = p0_0 + p0_1

	lower_bound = sum( [ [ Arange[0],crange[0],sigmarange[0]] for n in range(basis_size)], [])
	upper_bound = sum( [ [ Arange[1],crange[1],sigmarange[1]] for n in range(basis_size)], [])
	popt,pcov = curve_fit( model, x , y, p0 = p0,  maxfev = 10000000, bounds = (lower_bound, upper_bound))
	return popt
