# Script to hold various statistical functions
# Sections corresponding to different domains of stats will be clearly marked
import numpy as np
import matplotlib.pyplot as plt

#### Spectral functions (for frequency analysis) ####


#Periodogram. Note that there are many definitions. There is no windowing,
#averaging etc. Definition is same as found in "Simulation of fractional Brownian motion"
# by Ton Dieker

def periodogram(X,q):
    '''
    Input
    ---------
    observations X
    q an array of points we want to evaluate periodogram on
    ---------
    output
    ---------
    periodogram(spectral density estimate) values evaluated at q points
    '''
    N = np.size(X) # get the sample size
    Xbar = np.mean(X) # Get the average of data
    S = np.zeros(np.shape(q), dtype = "complex_") # preallocate S array
    for k in range(N):
        S[:] += (X[k] - Xbar) * np.exp(1j * k * q[:])
    return (1. / N) * np.abs(S) **2

#### Extreme value statistics ####

#The theory used to produce the code is based on "Modelling extremal events: For finance and insurance"
#by Kluppelberg, Embrechts and Mikosch

#Hill estimator
# estimates the tail exponent a for a cumulative probability distribution
# F(x) ~ 1 - x^-a over a range of sub-samples
def Hill_est(X, k):
    '''
    input
    ------
    data array X of size N
    k the size of the subsample
    ------
    output
    ------
    alpha the hill estimate of a in F(x) sim 1 - x^-a
    '''
    N = len(X) # get size of the data array
    Xord = np.flip(np.sort(X),0) # order stats X[0] > ... > X[N-1]
    Y = np.log(Xord) # Logarithm of the order statistics. Assume X > = 0
    alpha = 0
    for j in range(k):
        alpha += (Y[j] - Y[k-1]) / k #This expression will be inverted to get alpha
    return alpha**-1 # The hill estimator calculates alpha

def plot_Hill_est(X,kmin,kmax, kstep):
    '''
    Input
    ------
    data array X
    kmin the minimum subsample size
    kmax the maximum subsample size
    kstep the subsample increments
    ------
    Output
    ------
    A plot of the Hill estimator with confidence bands
    '''
    k_vals = np.arange(kmin, kmax, kstep) # get a list of subsample sizes
    Hill_vals = [ Hill_est(X, k) for k k_vals]
    # Get 95% confidence interval using consistency of Hill estimator
    Hill_plus = Hill_vals + 1.96 * Hill_vals / np.sqrt(k_vals)
    Hill_minus = Hill_vals - 1.96 * Hill_vals / np.sqrt(k_vals)
    #Produce the plot of Hill-estimates
    plt.plot(k_vals, Hill_vals, color = 'k')
    plt.fill_between(k_vals, Hill_plus, Hill_minus, color = 'b', alpha = 0.2)
    plt.title(' Hill estimate plot, 95% confidence')
    plt.xlabel('k')
    plt.ylabel(r'$\hat{\alpha}(k)$')
    plt.show()
    return

# Pickand estimator
def Pick_est(X, k):
    '''
    input
    ------
    data array X of size N
    k the size of the subsample
    ------
    output
    ------
    xi the Pickand estimate of xi = 1/a in F(x) sim 1 - x^-a
    '''
    N = len(X)
    Xord = np.flip(np.sort(X),0) # order the statistics as X[0] > ... > X[n]
    xi = np.log((Xord[k-1] - Xord[2 * k - 1]) / (Xord[2 * k - 1] - Xord[4 * k - 1]))/np.log(2)
    return xi

def plot_Pick_est(X, kmin, kmax, kstep):
    '''
    Input
    ------
    data array X
    kmin the minimum subsample size
    kmax the maximum subsample size
    kstep the subsample increments
    ------
    Output
    ------
    A plot of the Hill estimator with 95% confidence bands

    '''
    k_vals = np.arange(kmin, kmax, kstep) #list of subsample sizes
    xi = [Pic_est(X,k) for k in k_vals]
    #Get the 95% confidence bands on the Pickand estimator
    v = xi**2 * (2.**(2 * xi + 1) + 1) / (2 * (2.**xi - 1) * np.log(2) )**2 # variance in asymptotic normality condition
    c = 1.96 * np.sqrt(v / k) # to calculate upper and lower bound 
    xi_plus = xi + c
    xi_minus = xi - c
    #plot the xi estimate with confidence bands
    plt.plot(k_vals, xi, color = 'k')
    plt.fill_between(k_vals, xi_plus, xi_minus, color = 'b', alpha = 0.2)
    plt.title('Pickand estimate plot, 95% confidence')
    plt.xlabel('k')
    plt.yable(r' \hat{\xi}(k)')
    plt.show()
    return
#P-quantile estimate
# To do: include 95% CI, but this is not straightforward
def p_quant(X, p,k):
    Xord = np.flip(np.sort(X),0) # order the statistics as X[0] > ... > X[n]
    N = len(X) # get the sample siz
    alpha = Hill_est(X,k) # get the alpha estimate
    return Xord[k-1] * ((1. - p) * N / k)**(-1 / alpha) #estimate of p-quantile

#Pareto tail estimator
def Pareto_tail_est(X,k):
    N = len(X)
    Xord = np.flip(np.sort(X),0)
    alpha = Hill_est(X,k) #Shape parameter of Pareto
    def _F(x):
        return (float(k) / N) * (x / Xord[k-1])**(-alpha)
    return _F
#Empirical tail distribution function. F_bar = 1- F, so bar_ecdf = 1- ecdf
def bar_ecdf(X):
    Xord = np.sort(X) # sort the data
    N = len(X) # length of the sample
    def _bar_ecdf(v): # return the empirical survival function
        return 1 - (np.searchsorted(Xord, V, side = 'right')) / N
    return _bar_ecdf

####Quantile plots####
# The generalised inverses of the extreme value distributions( no translation)

#A standard Gumbel distribution
def Gumbel_inv(x):
    return -np.log(-np.log(x))

#A standard Frechet distribution
def Frechet_inv(x, alpha= alpha):
    return (-np.log(x))**(-1./alpha)

#A standard Weibull distribution
def Weibull_inv(x,alpha = alpha):
    return -(-np.log(x))**(1./alpha)

# Generalised inverse distribution
def GEV_inv(x, xi):# note this defn doesn't account xi=0 Gumbel case.
    if abs(xi) > 1e-10:
        return ((-np.log(x))**(-xi) - 1) / xi
    else:
        return -np.log(-np.log(x))

#QQplot for an arbitrarily supplied inverse distribution
def QQ_plot(X, F):
    '''
    input
    ----
    X the data set
    F the inverse of the 
    output
    ----
    QQplot
    '''
    N = len(X)
    Xord = np.sort(X)[::-1] # order stats X[0] > ... > X[n]
    pk = np.array([float((n-k + 0.5))/n for k in range(1, n+1)]) #points on which to evaluate inverse function
    Y = F(pk)
    plt.plot(Y, Xord, 'o')
    plt.show()
    return

    
    
