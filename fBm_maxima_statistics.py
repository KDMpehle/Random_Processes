'''
This script imports simulations from fBm maxima and looks at their distribution,
then estimates the parameters of the beta distribution that would best fit the
data.
'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from fBm import fBm, max_value
maxima_df = pd.read_csv("Hurst_01_95_50000_sample.csv")
Hursts = np.array(maxima_df["Hurst"].tolist())
#Hurst_inds = np.where(np.abs(Hursts- 0.95) < 1e-4)
#tau_m = np.array(maxima_df["max hit time"].tolist())[Hurst_inds]
H_vals = np.unique(Hursts) # List of unqiue Hurst values
N = len(H_vals) # sample size
param_vals = np.zeros((N,2)) # to hold beta distribution estimates.
KS_stats = np.zeros((N,2)) # to hold the D and p value of 2-sided KS.
print H_vals # check we get the right thing
#fig1, ax1 = plt.subplots(5,1,sharex= True)# axis for the first 5 Hurst parameters
#fig2, ax2 = plt.subplots(5,1,sharex=True) # axis for the last 5 Hurst parameters.
for k in range(N): # adapt this later for an arbitrary number of unique Hurst values
	H= H_vals[k]
	Hurst_inds = np.where(np.abs(Hursts-H) < 1e-4) # indices of maxima arrays corresponding to H
	tau_m =np.array(maxima_df["max hit time"].tolist())[Hurst_inds]
	sigma = np.var(tau_m,ddof=1)# unbiased sample variance.
	x_bar = np.mean(tau_m)# sample mean.
	#estiamte the beta parameters
	alpha, beta = x_bar*(x_bar*(1-x_bar)/sigma-1), (1-x_bar)*(x_bar*(1-x_bar)/sigma-1)
	param_vals[k,:] = alpha,beta # create parameters array. 
	tau_beta = np.random.beta(alpha,beta,N) #sample beta with MoM estimates
	#Perform a two-sided Kolmogorov-Smirnov test
	D,p = stats.ks_2samp(tau_m,tau_beta)
	KS_stats[k,:] = D,p
	#if k <5:
	#	ax = ax1[k]
	#else:
	#	ax= ax2[k-5]
	#ax.hist(tau_m,color="red",alpha = 0.5 ,label="H ={:.3f}".format(H))# histogram of simulated values
	#ax.hist(tau_beta,color="blue",alpha =0.5 ,label =" beta estimate") # histogram of estimated beta.
	#ax.set_yticks([])# no need to show number of samples
	#ax.legend(loc = "upper center")
print(param_vals)
print(KS_stats)
fig,ax = plt.subplots(1,2,sharey= True)
ax[0].plot(H_vals,param_vals[:,0],'x')
ax[0].set_title(r'$\hat{\alpha}$ against $H$')
ax[1].plot(H_vals,param_vals[:,1],'x')
ax[1].set_title(r'$\hat{\beta}$ against $H$')
fig.savefig("alpha_beta_against_H.pdf")
plt.show()
#fig.suptitle(r'$\tau$ for various Hurst parameters')
#plt.savefig("Max_hit_time_histograms.pdf") #save the beta distribution.
#fig1.savefig("max_hit_time_hist_low_H.pdf")
#fig2.savefig("max_hit_time_hist_high_H.pdf")
#plt.show() #show plot. 
