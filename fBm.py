''' This script holds various functions for
fractional Brownian motion and concepts related
to long range dependence'''
import numpy as np
import matplotlib.pyplot as plt
import scipy as scipy

def fBm(g,H=0.8,choice = 'fBm'):
    '''fGn is simulated using the Davies-Hearte method
    -g is the exponent of sample size: N= 2^g
    - H is the Hurst parameter, defaulted to H=0.8.
    '''
    def auto_cov(k, H=0.8): # the auto-covariance of fGn
        return (np.abs(k-1)**(2*H) -2*np.abs(k)**(2*H) + np.abs(k+1)**(2*H))/2
    N =2**g
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
    w = np.fft.fft(W, norm = "ortho") # The Fast Fourier transform
    #We return the first N elements for a sample of fGn.
    if choice == 'fBm':
        return np.cumsum(np.real(w[0:N])) # cumulative sum to get fBm from increment process.
    elif choice == 'fGn':
        return w[0:N] 
    else:
        print("Error: choice must be either 'fGn' or 'fBm'")
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
def max_hit(X,t):
    max_index = np.argmax(X) # Get the index of the maximum of the process
    return t[max_index] # return the time of attainment of maximum 
if __name__== "__main__":
    #Test out the periodogram method. 
    #Produce a fractional Gaussian noise H=1/2 for moment
    from scipy.stats import norm,linregress
    g=8
    N=2**g
    fGnoise= fBm(g,H=0.7,choice= 'fBm')
    plt.plot(np.array(range(len(fGnoise))),fGnoise)
    plt.show()
    '''q_vals = np.array([np.pi*k/N for k in range(1,N+1)])
    pgram = periodogram(fGnoise,q_vals)
    slope = linregress(np.log10(q_vals[0:800]),np.log10(pgram[0:800]))[0]
    print 'line slope is', slope,' and the estimated hurst parameter is H =',(1-slope)/2
    plt.loglog(q_vals,pgram,'o', ms =4)
    plt.show()'''
    #Whittle_est(fGnoise)
