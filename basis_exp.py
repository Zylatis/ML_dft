from scipy.optimize import minimize, curve_fit
import numpy as np
from matplotlib import pyplot as plt

n = 200
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

#~ def meanSSR( y_true, y_pred ):
	#~ return np.sum( (y_true - y_pred)**2)/n
	
basis_size = 5
x = np.linspace(-10,10,n)

y_true = model(x,1,1,1,1,4,1)

rng = np.random.RandomState(1)
y_data = y_true + 0.05*rng.rand(n)
Arange = [0,10]
crange = [-10,10]
sigmarange = [0.001,10]

lower_bound = sum( [  [Arange[0],crange[0],sigmarange[0]] for n in range(basis_size)], [])
upper_bound = sum( [ [ Arange[1],crange[1],sigmarange[1]] for n in range(basis_size)], [])


#~ popt,pcov = curve_fit(model,x, y_true, p0 = [1]*3*basis_size, bounds = (lower_bound, upper_bound) )
#~ plt.plot(x,model(x,*popt))

#~ plt.scatter(x,y_true)
#~ plt.scatter(x,y_data)
#~ plt.show()

with open('train_data/densities.dat', 'r') as ins:
    density_data = [[float(n) for n in line.split()] for line in ins]

dat = density_data[1]
n_points = len(dat)
r_space = np.linspace(-10,10,n_points)


popt,pcov = curve_fit(model,r_space, dat, p0 = [1]*3*basis_size, bounds = (lower_bound, upper_bound) )

plt.plot(r_space,dat)
plt.plot(x,model(x,*popt))
plt.show()
