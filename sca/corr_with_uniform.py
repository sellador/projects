# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 23:29:14 2014

@author: Seve
"""

import numpy as np
import matplotlib.pyplot as plt


m = 32;
N = 10e3;
gain = 10e-3
x = np.arange(0,32,1)
x = np.repeat(x, N)
x = x - x.mean()

t = 1000
t0 = 500
r = np.zeros(t)
for k in range(t):
    y = np.random.randn(m*N)
    if k == t0:
        y = np.random.randn(m*N) + x*gain
    r[k] = np.corrcoef(x,y)[0,1]


plt.plot(r)

#
#r_hat = 0
#c = np.zeros(32)
#for i in range(m):
#    c[i] = y[i*N:(i+1)*N].sum()
#    r_hat = r_hat + i*c[i]    
#    print("mean: {0} std: {1}".format(y[i*N:(i+1)*N].mean(), y[i*N:(i+1)*N].std()))
#r_hat_approx = np.sum(np.arange(0,32,1)*y.sum())
#print("r: {0} r_hat: {1} r_hat_approx: {2}".format(r, r_hat, r_hat_approx))