'''
This script gives code to find QQ plots for
the standard extreme value distributions
- Frechet
- Weibull
- Gumbel
'''
import numpy as np
import matplotlib.pyplot as plt
import time 
alpha = 2. # the shape parameter for the Gumbel
# The generalised inverses of the extreme value distributions
def Gumbel_inv(x):
    return -np.log(-np.log(x))
def Frechet_inv(x, alpha= alpha):
    return (-np.log(x))**(-1./alpha)
def Weibull_inv(x,alpha = alpha):
    return -(-np.log(x))**(1./alpha)
def GEV_inv(x, xi):# note this defn doesn't account xi=0 Gumbel case.
    return ((-np.log(x))**(-xi)-1)/xi
# Now define the QQ plots
def QQ_plot(X,F):
    deltak = 0.5
    gammak = 0. #delta, gamma are continuity corrections.
    #X is an array of observations, F a functional variable
    n = np.size(X) # get the sample size
    Xord = np.sort(X)[::-1]# order stats X[0] > ... > X[n]
    pk =np.array([float((n-k + 0.5))/n for k in range(1, n+1)])
    Y = F(pk)
    fig= plt.figure()
    plt.plot(Y,Xord,'o')
    return fig
fig, axarr = plt.subplots(2,2) # 4-panel subplot.
fig.subplots_adjust(hspace = .5)
n=200
U= np.random.random(n)
X1 = Gumbel_inv(U) #Gumbel data.
X2 = GEV_inv(U,0.3) # Frehcet data.
X3 = GEV_inv(U,-0.3) # Weibull data.
X4 = GEV_inv(U,0.7) # More Frechet data.
X1ord=  np.sort(X1)[::-1] #order the observations in decreasing order.
X2ord=  np.sort(X2)[::-1]
X3ord=  np.sort(X3)[::-1]
X4ord=  np.sort(X4)[::-1]
pk = np.array([float((n-k + 0.5))/n for k in range(1, n+1)])
Y = Gumbel_inv(pk)
axarr[0,0].plot(X1ord,Y,'o',ms =3)
axarr[0,0].set_title('(a)')
axarr[0,0].set_xlabel('Quantiles of Gumbel data')
axarr[0,1].plot(X2ord,Y,'o',ms =3)
axarr[0,1].set_title('(b)')
axarr[0,1].set_xlabel(r'Quantiles of Frechet $\xi =0.3$')
axarr[1,0].plot(X3ord,Y,'o', ms =3)
axarr[1,0].set_title('(c)')
axarr[1,0].set_xlabel(r'Quantiles of Weibull $\xi = -0.3$')
axarr[1,1].plot(X4ord,Y,'o', ms =3)
axarr[1,1].set_title('(d)')
axarr[1,1].set_xlabel(r'Quantiles of Frechet $\xi = 0.7$')
plt.savefig("qq_plot_panel.pdf")
plt.show()
'''
This next section of the code implements a ratio of maximum and sum method,
a method for detecting heavy tailed behaviour in the sample
'''
def Rn(X,p):
    if p <= 0 or np.size(X) < 1:
        print('error: p and the size of X must be positive')
        return
    Sn = np.sum(np.power(np.abs(X),p))
    Mn = np.max(np.power(np.abs(X),p))
    Rn = Mn/Sn
    if Sn < 1e-6:
        print(' Warning: Sn is small: procedure may be unstable')
    return Rn

# We look at the Rn ratio for a Pareto distribution f shape parameter 2
U = np.random.random(2500) # a larger sample of standard uniforms
def Pareto_inv(x,alpha = alpha):
    return (1-x)**(-1./alpha) - 1
X = Pareto_inv(U) # iid Pareto distributions.
fig, ax = plt.subplots(4, sharex= True)
p_vals = [1.,2, 2.5,3.]
for i in range(0,4):
    p = p_vals[i]
    Rn_vals= [Rn(X[0:n+1],p) for n in range(0,np.size(X)+10,10)]
    ax[i].plot(range(0,2510,10), Rn_vals,label = 'p=%s'%p)
    ax[i].set_ylabel(r'$R_n$')
    ax[i].legend()
ax[3].set_xlabel('n')
ax[0].set_title(r'$R_n$ method: The Pareto distribution with $\alpha$ =%s'%int(alpha))
plt.savefig('Pareto_alpha_2_max_sum.pdf')
print(' looks okay... see if a flatter graph is observed for the exponential')
mu =1. # the scale parameter of the exponential. 
def exp_inv(x,mu = mu):
    return -(1./mu)*np.log(x)
X = exp_inv(X)
fig, ax = plt.subplots(4, sharex= True)
for i in range(0,4):
    p=p_vals[i]
    Rn_vals = [Rn(X[0:n+1],p) for n in range(0,np.size(X)+10,10)]
    ax[i].plot(range(0,2510,10), Rn_vals, label= 'p=%s'%p)
    ax[i].legend()
    ax[i].set_ylabel(r'$R_n$')
ax[3].set_xlabel('n')
ax[0].set_title(r'$R_n$-method: The standard exponential')
plt.savefig('exponential_mu_1_max_sum.pdf')
plt.show()
