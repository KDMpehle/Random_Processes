'''
A script to explore some aspects of
Adler and stochastic Adler equations,
specifically using the Euler_maruyama method of
stochastic differential equations to numerically
solve the stochast Adler equation. 
'''
import numpy as np
import matplotlib.pyplot as plt
from gaussQ import gaussxw, gaussxwab
pi=np.pi
def Euler_maruyama(mu,s,T,N,x0, N_traj):
    '''
    A script to solve the SDE:
    dX = mu(X,t)*dt + sigma(X,t)*dW.
    The parameters are:
    -mu(Xt,t) the drift of the process
    - s(Xt,t) the volatility of the process
    -T the time horizon of the solution
    -N number of time steps
    -x0 the initial condition
    '''
    dt = T/N
    t_vals = np.arange(dt, T+dt,dt) # time steps from dt to T
    Xt = np.zeros((N_traj,N)) # array to hold the number of trajectories
    Xt[:,0] = x0
    dW = np.random.normal(0,np.sqrt(dt),(N_traj,N))# N_traj by N array of indept. standard BM increments
    for m,t in enumerate(t_vals[:-1]):
        dw = dW[:,m] # Brownian motion increment for each trajectory
        Xt[:,m+1] = Xt[:,m] + mu(Xt[:,m],t)*dt +s(Xt[:,m],t)*dw
    return Xt,t_vals

def Adler_stationary_density(theta,a,s):
    C=1.# WARNING: find the normalisation constant C.
    M = 50 # number of quadrature points to use in Gauss quadrature.
    x,w = gaussxwab(M,theta,theta+2*pi)
    P = 0
    for k in range(M):
        P += (1./C)*w[k]*np.exp((-a*(x[k]-theta)-(np.sin(x[k])-np.sin(theta)))/s**2)/(s**2)
    return P
if __name__== '__main__':
    # we produce a sample of stochastic Adler's as an example
    a=2.
    sigma = 1.4
    N_traj = 5
    T=10.
    N=1000
    x0 = 0
    def h(theta,t,a=a):
        return a + np.cos(theta)
    def sig(theta,t,sigma=sigma):
        return sigma*np.ones(np.shape(theta))
    solution_choice = 'trajectory'
    if solution_choice == 'stationary density':
        theta_vals = np.linspace(0,2*pi,50)
        P_vals=[Adler_stationary_density(theta,a,sigma) for theta in theta_vals]
        plt.plot(theta_vals,P_vals)
        plt.show()
    elif solution_choice == 'trajectory':
        Xt,t_vals = Euler_maruyama(h,sig,20,N,0.,1)
        plt.plot(t_vals,Xt.flatten())
        plt.show()
    
