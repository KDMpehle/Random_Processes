'''
In this script we provide some sample trajectories of the partial
maxima functional Mn = max{ X1,..., Xn} for a sequence of iid Xi.
We do this for the exponential and Cauchy cases.
'''
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
N = 500 # the number of iid rvs.
theta = 1. # the exponential scale parameter. 
def partial_maxima(Finv):
    # Here F is a functional argument: input the generalised inv of
    # the DF F
    U = np.random.random(N)
    X = Finv(U)
    Mn = [np.max(X[0:i]) for i in range(1,N)]
    return Mn
def exp_inv(x,theta = theta): #e
    return -(1./theta)*np.log(x)
def Cauchy_inv(x):
    return np.tan(np.pi*(x-0.5))
Mn1, Mn2 = partial_maxima(exp_inv), partial_maxima(Cauchy_inv)
f, ax = plt.subplots(2, sharex= True) # Subplots share the x-axis.
ax[0].step(range(1,N),Mn1)
ax[1].step(range(1,N), Mn2)
ax[0].set_title( r'$ M_n$ against $n$ ')
ax[1].set_xlabel(r'$n$')
plt.show()
