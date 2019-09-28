'''
In this python script we look at the distribution of the maximum
of fractional Brownian motion, for the report I wish to write
on my undegraduate summer research project
'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fBm import fBm, max_value
# parameters
g= 10 # N= pow(2,g) sample size of the fBm
N= 2**g # samplesize. 
H_vals = np.linspace(0.1,0.95,10) # simulations over a sample of Hurst parameters.
#H_vals = H_vals[::-1] # wish to reverse the list. 
maxima, max_time = [],[] # lists to hold each quantity
t= np.linspace(0,1,N+1) # time horizon. N+1 fBm as fGn had N points.
sample_size = 1000 # sample size for maxima and hitting time.
for H in H_vals:
    for k in range(sample_size):
        B = fBm(g,H,'fBm',1.) # simulate the fractional Brownian motion sample
        M,tau = max_value(B,t) # get maximum and time of max.
        maxima.append(M) # store quantities in a list.
        max_time.append(tau)
    #print "Hurst value",H," completed"
data = {'Hurst':np.repeat(H_vals,sample_size)
        ,'max':maxima, 'max hit time': max_time}
maxima_df = pd.DataFrame(data)
Hursts = np.array(maxima_df["Hurst"].tolist())
Hurst_inds = np.where(np.abs(Hursts -0.2) <1e-6)
print Hursts[Hurst_inds]
maxima_df.to_csv("Hurst_01_95_{:d}_sample.csv".format(sample_size))
