''' This script holds various functions for
fractional Brownian motion and concepts related
to long range dependence'''
import numpy as np
import matplotlib.pyplot as plt
import scipy as scipy
from scipy.special import gamma 
import warnings

def fBm(g,H=0.8,choice = 'fBm',T=1.):
    '''fGn is simulated using the Davies-Hearte method
    -g is the exponent of sample size: N= 2^g
    - H is the Hurst parameter, defaulted to H=0.8.
    -T is the time horizon for [0,T]
    '''
    def auto_cov(k, H=0.8): # the auto-covariance of fGn
        return (np.abs(k-1)**(2*H) -2*np.abs(k)**(2*H) + np.abs(k+1)**(2*H))/2
    N =2**g # number of simulation points
    mesh = T/N # the number of points in the mesh
    C = np.zeros((2*N,2*N)) # the Circulant matrix, preallocated.
    C[0,0:N] = auto_cov(np.arange(0,N),H)
    C[0,N+1]= 0
    C[0,N+1:2*N] =auto_cov( np.arange(N-1,0,-1),H) # First row
    # note in last step: is there an error in Diekker?
    for i in range(1,2*N):
        C[i,:] = np.roll(C[0,:],i)# not i-1 due to numpy indexing.
    # Spectrally decompose C.
    r  = C[0,:]
    L = (2*N)*np.fft.ifft(r) # Numpy DFT is inverse of Diekker's.
    V1,V2 = np.random.normal(0,1.,2*N),np.random.normal(0,1.,2*N)
    W=np.zeros(2*N,dtype=np.complex_)
    W[0]= np.sqrt(L[0]/(2*N))*V1[0]
    W[1:N] = np.sqrt(L[1:N]/(4*N))*(V1[1:N] +1j*V2[1:N])
    W[N] = np.sqrt(L[N]/(2*N))*V1[N]
    W[N+1:2*N] = np.sqrt(L[N+1:2*N]/(4*N))*(V1[N-1:0:-1]-1j*V2[N-1:0:-1])
    # We take the fast Fourier transform of the special vector W
    w = np.fft.fft(W, norm ="ortho") # Redo the algorithm remove "ortho".
    #We return the first N elements for a sample of fGn.
    if choice == 'fBm':
        sample = np.insert((mesh)**H*np.cumsum(np.real(w[0:N])),0,0) #insert 0 at t=0
    elif choice == 'fGn':
        sample = (mesh)**H*np.real(w[0:N]) # fGn: note self-similar scaling. 
    else:
        print("Error: choice must be either 'fGn' or 'fBm'")
    return sample # return the sample with no adjustment
def mBm(H,T =1., N = 100):
    '''
    A function to produce multifractional Brownian motion.

    H: The Hurst parameter, a deterministic function with range (0,1)
    T: time horizon, with t in [0,T]
    N: Number of increments
    '''
    dt = T/N # increment size
    tt = np.linspace(0,T, N+1) # time grid.
    def w(t,h): # weight function for the Riemann-Liouville integral.
        return np.sqrt((t**(2*h)-(t- dt)**(2*h))/(2*h*dt))/gamma(h+0.5)
    xi = np.random.randn(N) # standard normal for Riemann-Liouville int.
    dW = xi*np.sqrt(dt)# Wiener integral approximation increments.
    hh = H(tt) # grid of Hurst parameters
    B = [0] # array to hold multifractional Brownian motion.
    for k in range(1,N+1):
        weight = w(tt[1:k+1],hh[k]) # list of relevant weights.
        B_tk = sum(dW[:k-1]*weight[0:k-1][::-1]) # returns the latest mBm value.
        B.append(B_tk)
    return B

#To produce a fractional Brownian motion bridge
def f_bridge(b,H,T,g):
    '''Generate an fBm bridge from 0 to b on [0,T]
    - b is the desired end point
    -T is the time horizon
    -H is the Hurst index
    -g gives sample size N = 2**g
    fBm bridge generated with:
    b*G(T,t)/G(T,T) + [X_t - X_T*G(T,t)/G(T,T)]
    '''
    N= 2**g # sample size
    t = np.linspace(0,T,N+1) # t-values
    X_t = fBm(g,H,'fBm',T) # produce fractional Brownian motion.
    B_t = X_t -(t**(2*H) +T **(2*H) -np.abs(t-T)**(2*H))*(X_t[-1]-b)/(2*T**(2*H))
    return B_t
# Gaussian field 2-D simulation with circulant matrices
def Gauss_field(R,n,m):
    '''
    Simulate a 2-D stationary Gaussian field with circulant matrix method.
    -R is the covaraince function of the Gaussian process
    -nx,ny is the number of x and y grid points.

    Code vectorises the Rows and column matrices construction for speed.
    May need to be careful about functions used. For e.g, a function
    constructed using a quadratic form needs to be defined as an explicit
    sum instead of through the matrix product. 
    '''
    tx,ty = np.array(range(n),float),np.array(range(m),float)# grid for the field.
    Row, Col= np.zeros((m,n)), np.zeros((m,n)) # set up row and column matrices
    Row = R(tx[None,:] - tx[0],ty[:,None]-ty[0])
    Col = R(-tx[None,:] + tx[0],ty[:,None] - ty[0])
    BlkCirc_rw = np.vstack(
        [np.hstack([Row,Col[:,-1:0:-1]]),
        np.hstack([Col[-1:0:-1,:],Row[-1:0:-1,-1:0:-1]])]) # construct block circulant matrix
    lam = np.real(np.fft.fft2(BlkCirc_rw))/((2*m-1)*(2*n-1)) # eigenvalues
    neg_vals = lam[lam <0]
    if len(lam[lam <0]) == 0: #If there are no negative values, continue.
        pass # No need to adjust eigenvalue matrix. 
    else:
        if np.absolute(np.min(neg_vals)) > 10e-15:
            raise ValueError(" Could not find a positive definite embedding  ")
        else:
            lam[ lam < 0] = 0
    lam = np.sqrt(lam) # component-wise squareroot.
    Z =( np.random.randn(2*m-1,2*n-1)
         + 1j*np.random.randn(2*m-1,2*n-1) ) # add a standard normal complex vector
    F = np.fft.fft2(lam*Z)
    F = F[:m,:n]
    field1, field2 = np.real(F), np.imag(F)
    return field1,field2
#To interpolate a time series with a series of Brownian motions
def BM_stochastic_interpolate(x,ts,k):
    """
    Interpolate between grid points using Brownian motion
    idea is to produce stochastic interpolatory model of
    channel floor height.

    x: array of time series points
    ts: time series associated with x
    k: number of points to interpolate between each data points.
    """
    if len(ts) != len(x):
        raise ValueError("time series and values must be equal")
    if not isinstance(k,int) or k <=0:
        raise ValueError(" number of bridge points must be a positive integer")
    dt = 1.0*(ts[1] - ts[0]) # get the time increments. dt must be a constant.
    xi = np.linspace(dt/k,dt,k-1) # time points between [0,dt]
    Wt = np.cumsum(np.sqrt(dt/(k-1))*np.random.randn(k-1)) # Brownian motion
    #Wt = np.insert(Wt,0,0) # insert initial condition of the Brownian motion.
    interpolate = [x[0]]
    N = len(x) # length of the original time series.
    for i in range(N-1):
        #Wt = Wt + x[i] # start point of the Brownian motion.
        Wt = np.cumsum(np.sqrt(dt/(k-1))*np.random.randn(k-1)) + x[i] # Brownian motion
        Bridge_segment = Wt - xi*(Wt[-1] - x[i+1])/dt
        interpolate = np.hstack([interpolate,Bridge_segment]) # add next segement of interpolate.
    ts2 = np.linspace(ts[0],ts[-1], (k-2)*(n-1) + n) #refined grid of interpolate.  
    return interpolate,ts2# return with the initial condition.
# For periodogram analysis.
def periodogram(X,q):
    '''
    Define the periodgram where
    X is an array and q the spectral variable.
    '''
    N = np.size(X) # the size of the sample. 
    Xbar = np.mean(X) # Xbar, average of data.
    S = 0
    for k in range(0,N):
        S+= (X[k]-Xbar)*np.exp(1j*k*q)
    return (1./N)*np.abs(S)**2
# An approximation to fGn's spectral density
def f_H(q,H=0.8):
    def aplus(k,q):
        return 2*np.pi*k + q
    def aminus(k,q):
        return 2*np.pi*k -q
    def B(q,H): # an approximation to B(H) in fGn spec. density
        B3 =0
        for j in range(1,4):
            B3 += aplus(j,q)**(-2*H-1) +aminus(j,q)**(-2*H-1)
        B3+= (aplus(3,q)**(-2*H) + aminus(3,q)**(-2*H)
              +aplus(4,q)**(-2*H) + aminus(4,q)**(-2*H))/(8*H*np.pi)
        return (1.0002 -0.000134*q)*(B3 -2**(-7.65*H-7.4))
    B3 = B(q,H)
    return 2*np.sin(np.pi*H)*scipy.special.gamma(2*H+1)*(1-np.cos(q))*(np.absolute(q)**(-2*H-1) + B3)
              
# For Whittle's method
def Whittle_est(X):
    N = np.size(X)
    def Q_tilde(H):
        Q= 0
        L = np.array([2*np.pi*k/N for k in range(1,(N/2)+1)])
        for k in range(0,(N/2)):
            Q+= periodogram(X,L[k])/f_H(L[k],H)
        return Q
    # Here we minimise Q(H) on the specified grid L
    H0 = 0.5
    print(scipy.optimize.minimize(Q_tilde,H0))
    
    '''
    The Whittle estimator involves minimising the function
    Q(H) defined in Diekker's thesis under section 3.1.8
    '''
    
''' Come back to this: Needs a proper analysis.
# This is for the R/S analysis
def RS_hurst(X,K):
    N = np.size(X)
    M = int(np.floor(N/k))
    def S(t,r):
        s1,s2=0
        for j in range(0,r):
            s2+= (1./r)*X[t+j]
            s1+= (1.r)*X[t+j]**2
        s= s1-s2**2
        return np.sqrt(s)
    def R(t,r):
        def W(t,k):
            s1,s2 = 0
            for j in range(k):
                s1+= X[t+j]
            for j in range(r):
                s2+=(1./r)*X[t+j]
            return s1-k*s2
        return (np.max([W(t,k) for k in range(1,r+1)])
                - np.min([W(t,k) for k in range(1,r+1)]))
    t_vals = np.linspace(0,M*(K-1),K) # The t-values
    r0 = 10 # base value of r
    r_vals = [r**k for k in range(1,10)]
    RS_vals = R(t_vals,r_vals)
 '''
#Now we define some time functionals of the fractional Brownian motion
'''Let us define some functionals associated with stochastic processes
- The first is the time at which the process hits its maximum'''
def max_value(X,t):
    max_index = np.argmax(X) # Get the index of the maximum of the process
    return X[max_index],t[max_index] # return the time of attainment of maximum

test = "mBm"
if __name__== "__main__" and test =="fBm":
    #Test out the periodogram method. 
    #Produce a fractional Gaussian noise H=1/2 for moment
    from scipy.stats import norm,linregress
    g=8
    N=2**g
    #sample= fBm(g,H=0.8,choice= 'fBm')
    sample = f_bridge(0.,0.1,3.,8)
    t= np.linspace(0,1.,N+1)
    plt.plot(t,sample)
    plt.show()
    
if __name__ == "__main__" and test == "Gauss field":
    def R1(x,y,l=1./np.sqrt(10)):
        A = np.array([[2,1],[1,2]])
        s =( (x/15)**2*A[1,1] + (A[0,1] +A[1,0])*(x/15)*(y/50)
             + (y/50)**2*A[0,0])
        Rho = np.exp(-np.sqrt(s))
        return Rho
    from scipy.special import kv
    def R2(x,y, v = 0.25):
        A = np.array([[3,1],[1,2]])
        s =np.sqrt(A[1,1]*(x/15)**2 + 2*A[0,1]*(x/15)*(y/50)+ A[0,0]*(y/50)**2)
        Rho = np.ones(s.shape)
        Rho[s>0] =  2**(1-v)*(np.sqrt(2*v)*s[s>0])**v*kv(v,np.sqrt(2*v)*s[s>0])/gamma(v)
        return Rho
    field1,field2 = Gauss_field(R2,512,512)
    plt.imshow(field1)
    plt.show()
if __name__ == "__main__" and test == "mBm":
    def H(t):
        return 0.5 + 0.45*np.sin(8*np.pi*t)
    s = r'$H(t) = 0.5 + 0.45 \sin(8\pi t)$'
    T = 1.
    N = 400
    t_vals = np.linspace(0,T,N+1)# time grid
    sample = mBm(H,T,N)
    plt.plot(t_vals, sample)
    plt.title(s)
    plt.show()
