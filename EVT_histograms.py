'''
A script to calculate the distribution of the extremal values
of various distributions under appropriate scaling. The possible
distribution which the scaled maxima can take on in classical
extreme value theory are:
- Frechet
-Weibull
-Gumbel

In what follows Xi denotes a random variable and
Mn = max{X1,...,Xn} their partial maxima.
We suppose that we only have access to simulating
a standard uniform random variable.
'''
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
M = 1000 # the number of samples
N = 10000 # number of iid rvs for each sample.

# First let us look at the exponential case
theta = 1. # The scale parameter.

U = np.random.random((M,N)) # M samples of maxima of N Xi
X = -(1/theta)*np.log(U) # This simulates iid exponentials.
Mn =  np.amax(X,axis= 1)
Mn -= np.log(N)  #the scaling for exponential.
plt.hist(Mn,bins = 15)
plt.title(' Maxima for exponential RVs')
plt.xlabel(r'$ M_n - \ln n$', fontsize = 14) 
plt.show()
print ('That looks OK for a Gumbel distribution.\
we move onto the Pareto distribution.')
print('Hmm, interesting... on to the standard Cauchy dist')
X = np.tan(np.pi*(U-0.5))
Mn = np.amax(X,axis=1)
Mn=np.pi*Mn*(N**-1)
plt.hist(Mn,bins=15)
plt.title(' Maxima of Cauchy RVs')
plt.xlabel(r'$\pi M_nn^{-1}$',fontsize = 14)
plt.show()
print('Again, looks like the distribution is too heavy-tailed.')
print(' What about the uniform distribution itself?')
Mn = np.amax(U,axis =1)
Mn=  N*Mn - N
plt.hist(Mn, bins = 15)
plt.title(' maxima of uniform RVs.')
plt.xlabel(r'$nM_n - n$', fontsize = 14) 
plt.show()
print('This last has the shape of a Weibull type distribution.')
