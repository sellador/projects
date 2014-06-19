# -*- coding: utf-8 -*-
"""
Created on Mon Feb 03 23:22:36 2014

@author: Seve
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import sklearn.metrics as skm

# Load input data and traces

matfile = sio.loadmat('C:\Users\Seve\Documents\projects\sca\WS2\WS2.mat')
inputs = matfile['inputs']
traces = matfile['traces']
sbox = matfile['SubBytes']
HW = matfile['byte_Hamming_weight']
# Plot the mean of the traces for sanity's sake
meanTR = np.mean(traces,0)
#plt.plot(meanTR)
#plt.show()

# Calculate the 8-bit HW of the input data
inHW = HW[0, inputs]

# Calculate the column-wise standard deviation of the traces
stdTR = np.std(traces,0)

# Standardize the data
TR = (traces - meanTR)/stdTR

# Calculate the dot product between the input HW and the standardized traces

r = np.dot(np.transpose(inHW), TR)

for i in range(r.shape[0]):
    plt.plot(r[i,:])
    
plt.show()
#plt.plot(r)
#plt.show()

best_key = np.zeros((16,1))
for target_byte in range(16):
    r_sb = np.zeros((256, TR.shape[1]))
    for key_guess in range(256):
        addOut = inputs[:, target_byte]^key_guess
    
        sboxOut = sbox[0, addOut]
        sbHW = HW[0, sboxOut]
        r_sb[key_guess,:] = np.dot(np.transpose(sbHW), TR)
        #plt.plot(r_sb[key_guess,:])

    #plt.show()
    best_key[target_byte,0] = np.argmax(np.max(r_sb, 1), 0)
    

# Now try using the mutual information distinguisher


t_left = 28000
t_right = 30000
best_key_mi = np.zeros((16,1))

for target_byte in range(16):
    score = np.zeros((256,t_right-t_left))
    for key_guess in range(256):
        addOut = inputs[:, target_byte]^key_guess
        sboxOut = sbox[0, addOut]
        sbHW = HW[0, sboxOut]
        for t in np.arange(t_left,t_right,1):  
            score[key_guess,t-t_left] = skm.mutual_info_score(TR[:,t], sbHW) 
            #plt.plot(r_sb[key_guess,:])

            #plt.show()
    best_key_mi[target_byte,0] = np.argmax(np.max(score, 1), 0)
   

plt.plot(score[best_key[target_byte,0],:])
plt.show()

    