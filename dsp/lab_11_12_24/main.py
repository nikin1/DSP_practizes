import numpy as np
import matplotlib.pyplot as plt

from scipy import signal
from scipy.signal import kaiserord, lfilter, firwin, freqz
from scipy import fftpack


f_s = 8 * 10 ** 3
f_1 = 1000
f_2 = 2300

T_s = 1 / f_s
t = np.arange(0, 255) * T_s

N = 37
m = np.arange(0, 20, 1)
w_hem = 0.54 - 0.46 * np.cos((2 * np.pi * m)/(N - 1))

s = 5 * np.cos(2 * np.pi * f_1 * t) + np.sin(2 * np.pi * f_2 * t)

s3 = s[2::3]
sp3 = fftpack.fft(s3)
freqs3 = np.arange(0, f_s/3, f_s/3/len(s3))

sp = fftpack.fft(s)
freqs = np.arange(0, f_s, f_s/len(s))

plt.figure(1)
plt.plot(freqs, np.abs(sp))
plt.xlabel('Частота в герцах [Hz]')
plt.ylabel('Модуль спектра')

plt.figure(2)
plt.plot(freqs3, np.abs(sp3))
plt.xlabel('Частота в герцах [Hz]')
plt.ylabel('Модуль спектра')

plt.show()


