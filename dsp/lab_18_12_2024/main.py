import numpy as np 
import matplotlib.pyplot as plt 
from scipy.signal import butter, lfilter, freqz 
 
def butter_lowpass(cutoff, fs, order=37): 
    nyq = 0.5 * fs 
    normal_cutoff = cutoff / nyq 
    b, a = butter(order, normal_cutoff, btype='low', analog=False) 
    return b, a 
 
def lowpass_filter(data, cutoff, fs, order=5): 
    b, a = butter_lowpass(cutoff, fs, order=order) 
    y = lfilter(b, a, data) 
    return y 
 
 
fs = 8 * (10 ** 3) 
 
L = 3 
M = 8 
 
fpass = 1300 
fstop = 2000 
 
# deltaF = (fstop - fpass) / fs 
 
# N = 3.3 / deltaF # 37 
 
t = np.arange(0,0.04,1/fs) 
 
St = 5 * np.cos(2 * np.pi * 1000 * t) + np.sin(2 * np.pi * 2500 * t) 
 
fft_signal = np.fft.fft(St) 
 
fft_freq = np.fft.fftfreq(len(St),1/fs) 
 
plt.subplot(2,1,1) 
plt.plot(St) 
plt.subplot(2,1,2) 
plt.plot(fft_freq[:len(St)//2],np.abs(fft_signal)[:len(St)//2]) 
plt.show() 
 
#
# ФНЧ
#

filtered_signal = lowpass_filter(St,fpass, fs)

#
# ФНЧ
# 
 
StDec = filtered_signal[::M] 
 
fft_signal_Dec = np.fft.fft(StDec) 
 
fft_freq_Dec = np.fft.fftfreq(len(StDec),1/(fs/M)) 
 
omega = 2 * np.pi * 1650 / fs

n = np.arange(-18,19)
 
h = np.sin(omega * n) / np.pi * n 
 
hw = h * np.hamming(37) 
 
print(np.hamming(37)) 
 
plt.subplot(3,1,1) 
plt.plot(StDec) 
plt.subplot(3,1,2) 
plt.plot(fft_freq_Dec[:len(StDec)//2],np.abs(fft_signal_Dec)[:len(StDec)//2]) 
plt.subplot(3,1,3) 
plt.plot(hw) 
 
plt.show()

