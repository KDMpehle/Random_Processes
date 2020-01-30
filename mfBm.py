# Test script for simulating multifractional Brownian motion
# according to method of Rambaldi and Pinazza
# method based on Holmgren-Riemann-Liouville fractional integral
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
# some parameters
T = 1. # time horizon
N = 500 # number of grid points.
dt = T/N # Increment sizes.
def mBm(H,T=T,N=N):
	'''
	Function to produce multifractional Brownian motion.
	H: the Hurst parameter, a deterministic function.
	T: time horizon, should I just set to 1.?
	N number of increments.
	'''
	dt = T/N# increment size
	tt = np.linspace(0,T,N+1) # time grid.
	hh = H(tt) # grid of Hurst parameters.
	def w(t,h):
                return np.sqrt((t**(2*h)-(t- dt)**(2*h))/(2*h*dt))/gamma(h+0.5)
        #w = np.sqrt((tt**(2*hh)-(tt -dt)**(2*hh))/(2*hh*dt))/gamma(hh+0.5) # weights
        np.random.seed(0)
	xi = np.random.randn(N) #standard normals for Riemann-Liouville integral
	B = [0]
	coefs = [(g/np.sqrt(dt))*np.sqrt(dt) for g in xi]
	for k in range(1,N+1):
                weights = w(tt[1:k+1],hh[k])
                #seq = [coefs[i-1]*weights[k-i] for i in range(1,k+1)]
                seq =coefs[:k-1]*weights[0:k-1][::-1] 
                B.append(sum(seq))
	return B
#example: sinusoidal hurst function
def H(t):
	return 0.6*t+0.3
sample = mBm(H,T,N)
plt.plot(np.linspace(0,T,N+1),sample)
plt.show()

