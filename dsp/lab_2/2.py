import numpy as np
import matplotlib.pyplot as plt


#2 Моделирование гармонического колебания c различ фазами

t = np.linspace(0, 1, 100)
A = 5
f = 3
phi = [0, 90, 180, 270]

#plt.title('Моделирование гармонического колебания c различными фазами\n')
for i in range(len(phi)):
    x = A * np.sin(2 * np.pi * f * t + phi[i] * np.pi/180)
    plt.subplot(2, 2, i+1)
    plt.plot(t, x)
    plt.xlabel('Time (t)')
    
    #plt.title('$\Phi ={}^\circ$'.format(phi[i]))
#plt.ylabel('Amplitude (V)')