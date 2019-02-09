import numpy as np
import random
from scipy.optimize import minimize, curve_fit, basinhopping
from scipy.signal import find_peaks
import matplotlib
matplotlib.use('Agg')	
import matplotlib.pyplot as plt

# import numba as nb #import njit, prange

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



# @nb.jit()
def cost_fn( x0, *params ):
	x_grid = params[0]
	y_actual = params[1]
	y_pred = model( x_grid, *x0 )
	cost = np.sum((y_pred - y_actual)**2)
	return cost


# @nb.jit()
def opt_stochastic(args):
	x, y_actual, i, basis_size = args
	params = (x,y_actual)
	
	# p0 = [10.]*12
	peaks, peak_heights = find_peaks(y_actual, prominence = 0.01)
	p0_0 = sum( [ [ 0.1 , peaks[n], 10 ] for n in range(len(peaks))], [])
	p0_1 = []
	if len(peaks)<basis_size:
		p0_1 = sum( [ [0.1 , np.mean(peaks), 1] for n in range(basis_size - len(peaks))], [])
	p0 = p0_0 + p0_1
	fit = basinhopping(cost_fn, p0, minimizer_kwargs={"args" : params})
	bf_pars = fit['x']

	# plt.plot(x, y_actual )
	# plt.plot(x,  model( x,*bf_pars ))
	# plt.savefig( "../train_data/plots/" +str(i) + ".png" )
	# plt.clf()
	return fit['x'] #, len(peaks)


# @nb.jit()
# def do_fit( densities ):
# 	xgrid = np.linspace(0,N,N).astype(np.int32)
# 	# train_coeffs = np.zeros(N).astype(np.int32)

# 	basis_size = 3
# 	bf_pars, n_peaks = basis.test( xgrid, soln[0]	, basis_size )
# 	return 0