# This is a script to hold various functions used in statistics
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sts 

def mk_test(x, alpha = 0.05):
    '''
    This function implements the Mann-Kendall test for upard or downard trends
    in a statistical time series.
    ----input----
    - x: a vector containing a collection of time point observations.
    - alpha: the statistical significance level.

    ----output----
    - conclusion: A string giving the conclusion of statistical test.
    - H: a true or false variable about the null hypothesis
    - p: the p value obtained in the statistical test.
    - Z: the Mann-Kendall test statistic.

    test adapted from site:https://vsp.pnnl.gov/help/Vsample/Design_Trend_Mann_Kendall.htm

    '''
    N = len(x) # the length of the time series.
    s= 0 #s= # of postive differences minus # negative differences
    for k in range(1,N):
        for j in range(k+1,n+1):
            s += np.sign(x[j] - x[k])
    X = np.unique(x) # the unique data points
    g = len(X) # the number of tied groups
    if g == 1: # no tied groups in data
         var_s = N*(N-1)*(2*N + 5)/18
    else:
        var_s = N*(N-1)*(2*N+5) # initial variance
        for p in range(g):
            inds = (np.absolute(x - X[p]) < 1e-6).astype(np.int)# 1 where equaility is true
            tp = sum(inds) # the size of the pth tied group
            var_s -= tp*(tp-1)*(2*tp + 5)/18 # correct variance for ties.
    #compute the Mann-Kendall test statistic, Z:
    if s >0:
        Z = (s-1)/np.sqrt(var_s)
    elif s < 0:
        Z = (s+1)/np.sqrt(var_s)
    
    else:
        Z = 0 # if S == 0

    p = 2*(1- sts.norm.cdf(np.abs(Z))) #two tailed test
    H = np.abs(Z) > sts.norm.ppf(1- alpha/2)

    if (Z < 0) and H:
        conconclusion = "There is an increasing trend"
    elif (Z >0) and H:
        conclusion = "There is decreasing trend."
    else:
        conclusion = "There is no trend"
    return conclusion, H, p, Z
# A script to implement a principle component analysis (simplest form)
def PCA(x):
    '''
    A script to perform the principal component analysis as explained in
    Quantitative risk management by Embrechts et al.
    ----input----
    - x: an N-dimensional vector of data points

    ----output-----
    - Y: a N-dimensional vector of the principle components.
    '''
    Sigma = np.cov(x) # get sample covariance of the data.
    mu = np.mean(x) # get the sample mean.
    W, Gamma= np.linalg.eig(Sigma) # spectral decomp of covariance
    L = np.diag(np.sort(W)[::-1]) # diag(L1,L2,..,Ld), L1> L2>..>Ld
    Y = np.dot(Gamma.T,x- mu) # rotate the centered data
    return Y
if __name__ == "__name__":
    pass # here I will put a test case.
