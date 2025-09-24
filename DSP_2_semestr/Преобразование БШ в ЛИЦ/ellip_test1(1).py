# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 13:24:45 2025

@author: Admin
"""

 
 
import matplotlib.pyplot as plt
import numpy as np
import pylab
import scipy
import scipy.signal 
fc1 = 15
fc2 = 40
fs = 500
[b,a] = scipy.signal.ellip(6,3,50,fc1/fs );
 
 
 
fig = plt.figure()
plt.title('Digital filter frequency response')
ax1 = fig.add_subplot(111)
h,w = scipy.signal.freqz(b, a)
plt.semilogy(h, np.abs(w), 'b')
plt.semilogy(h, abs(w), 'b')
plt.ylabel('Amplitude (dB)', color='b')
plt.xlabel('Frequency (rad/sample)')
t = np.linspace(0, 1, 1000, False)  # 1 second
sig = np.sin(2*np.pi*10*t) + np.sin(2*np.pi*20*t)
f_sig = scipy.signal.lfilter(b, a, sig)