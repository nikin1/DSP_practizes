import numpy as np
def DFT_slow(x):
    """Compute the discrete Fourier Transform of the 1D array x"""
    x = np.asarray(x, dtype=float)
    
    N = x.shape[0]
    # print("N: ", N)
    n = np.arange(N)
    # print("n: ", n)
    k = n.reshape((N, 1))
    # print("k: ", k)
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)

x = np.random.random(1024)
np.allclose(DFT_slow(x), np.fft.fft(x))
print(x)
y = DFT_slow(x)
# print(y)