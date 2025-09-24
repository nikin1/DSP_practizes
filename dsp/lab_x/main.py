from scipy.fftpack import fft, ifft , fftshift
import numpy as np
import matplotlib.pyplot as plt

fc = 10 # Частота косинуса

fs = 32 * fc # частота дискретизации , избыточная
t  = np.arange (0, 2, 1 / fs) # длительность сигнала 2 с

x = np.cos(2 * np.pi * fc * t) # формирование временного сигнала

plt.figure(1)
plt.plot(t, x)

#plt.stem(t,x) # для отображения временных отсчетов сигнала ,
# выбрать длительность 0.2 сек

plt.xlabel('$t=nT_s$')
plt.ylabel('$x [n]$')
# Далее вычисляется ДПФ длиной N = 256 точек в интервале частот 0 − fs.

N = 256 # количество точек ДПФ

X = fft(x, N)/N # вычисление ДПФ и нормирование на N


X_n = [1, 0, 2, -1, 0, 0, 3, 0]
N = len(X_n)
# X_k = X_n * np.e ** (-j* n)
X_k = []
for k in range(N):
    X_k_i = 0
    for n in range(len(X_n)):
        # X_k_i = X_n(n) * np.e ** (-j * n * k * 2 * np.pi / N)
        X_k_i += X_n(n) * (np.cos(-j * n * k * 2 * np.pi / N) * -j* np.sin(-j * n * k * 2 * np.pi / N))
        
    # print(k, ":", X_k_i)