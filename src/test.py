#~ from modules import basis
#~ import numpy as np	
#~ from matplotlib import pyplot as plt

#~ with open('../train_data/densities.dat', 'r') as ins:
    #~ density_data = [[float(n) for n in line.split()] for line in ins]

#~ dat = density_data[1]
#~ n_points = len(dat)
#~ x = np.linspace(-10,10,n_points)

#~ bf_pars = basis.gaussian_exp( x, dat, 5 )
#~ print bf_pars
#~ plt.plot( x, dat )
#~ plt.plot( x, basis.model( x,*bf_pars ) )
#~ plt.show()
