from __future__ import division
import pandas as pd
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt

L = 100
N = 1000
dx = L/(N-1)
x = np.linspace(0,L,N)
H = diags([1., -2., 1.], [-1,0, 1], shape=(N, N))/dx**2

#~ potential = [ (r-N/2)**2 for r in range(0, N) ]
potential = [ -1./np.sqrt((r-N/2)**2 + 4) for r in range(0, N) ]
V = diags(potential, 0,  shape=(N, N))

n_levels = 2
vals, vecs = eigsh(-0.5*H + V, which='SA', k = n_levels)
#~ print vals[3]-vals[2]
#~ print vals[2]-vals[1]

#~ print np.sqrt(np.dot(vecs,vecs))
#~ print np.round(vals, 6)
#~ print np.round([(n*np.pi/L)**2/2 for n in range(1,6)], 6)
plt.plot(x,vecs[:,n_levels-1])
plt.plot(x,vecs[:,n_levels-2])
#~ potential = [ -.1/np.abs(r-N/2+0.1) for r in range(0, N) ]
plt.plot(x,potential)
plt.show()

