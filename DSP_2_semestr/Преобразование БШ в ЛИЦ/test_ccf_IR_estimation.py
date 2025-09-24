# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 04:05:37 2025

@author: Admin
"""

import numpy as np
import matplotlib.pyplot as plt


import scipy.signal as sig

N = 10000  # number of samples for input signal
K = 50  # limit for lags in ACF

# generate input signal
# normally distributed (zero-mean, unit-variance) white noise
np.random.seed(5)
x = np.random.normal(size=N)
# impulse response of the system
#h = np.concatenate((np.zeros(10), sig.windows.triang(10), np.zeros(10)))
# output signal by convolution
h = np.array([1, 0.7, 0.5, 0.3, 0.1, 0.07])
y = np.convolve(h, x, mode="full")


def compute_correlation_function(x, y):
    """Compute correlation function/kappa."""
    N, M = len(x), len(y)
    ccf = 1 / N * np.correlate(x, y, mode="full")
    kappa = np.arange(-M + 1, N)

    return ccf, kappa


def plot_correlation_function(cf, kappa):
    """Plot correlation function."""
    plt.stem(kappa, cf)
    plt.xlabel(r"$\kappa$")
    plt.axis([-K, K, -0.2, 1.1 * max(cf)])


# compute correlation functions
acfx, kappax = compute_correlation_function(x, x)
acfy, kappay = compute_correlation_function(y, y)
ccfyx, kappayx = compute_correlation_function(y, x)

# plot ACFs and CCF
plt.rc("figure", figsize=(10, 3))
plt.figure()
plot_correlation_function(acfx, kappax)
plt.title("Estimated ACF of input signal")
plt.ylabel(r"$\hat{\varphi}_{xx}[\kappa]$")

plt.figure()
plot_correlation_function(acfy, kappay)
plt.title("Estimated ACF of output signal")
plt.ylabel(r"$\hat{\varphi}_{yy}[\kappa]$")

plt.figure()
plot_correlation_function(ccfyx, kappayx)
plt.plot(np.arange(len(h)), h, "g-")
plt.title("Estimated and true impulse response")
plt.ylabel(r"$\hat{\varphi}_{yx}[k]$, $h[k]$");

