import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import kaiserord, lfilter, firwin, freqz
from scipy import fftpack
 
f1= 80 # выбрать частоту 
f2= 150 #выбрать частоту 
Ts=1e-3
fs=1/Ts
t = np.arange(0, 100)*Ts
s =  np.cos(f1*t*(2*np.pi)) + np.cos(f2*t*(2*np.pi))
 


plt.figure(2)
sp = fftpack.fft(s)

freqs=np.arange(0,fs,fs/len(s))

plt.plot(freqs, np.abs(sp))
plt.xlabel('Частота в герцах [Hz]')
plt.ylabel('Модуль спектра')

fc = 110
wc = fc*2*np.pi/fs
M = 25
n=np.arange(-M,M+1) 
print("n: ", n)

h=np.sin(wc*n)/(np.pi*n)
h[M]=wc/np.pi
plt.figure(3) #Импульсная характеристика фильтра h(n)
plt.stem(h)
plt.xlabel('Импульсная характеристика фильтра h(n)')
 

plt.figure(4)
w, hf = freqz(h, fs, whole=True) #Частотная  характеристика фильтра H(jw)
plt.plot(w,20*np.log(np.abs(hf))) 
plt.xlabel(' Частотная  характеристика фильтра H(jw)')



y=np.convolve(s,h) # Вычисление сигнала на выходе фильтра через свертку 

plt.figure(5)
yf = fftpack.fft(y)

freqs=np.arange(0,fs,fs/len(yf))
plt.plot(freqs, np.abs(yf))
plt.xlabel('Частота в герцах [Hz]')
plt.ylabel('Модуль спектра сигнала на выходе фильтра') 
 

 
 






