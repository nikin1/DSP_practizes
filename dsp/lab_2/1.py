import numpy as np
import matplotlib.pyplot as plt


#1 Моделирование гармонического колебания
# x(t) = A*cos(2pi*f*t + ph)
t = np.linspace(0, 2.5, 250)
A1 = 3
A2 = 5
A3 = 1

f1 = 2
f2 = 4
f3 = 6

ph = 0

x1 = A1 * np.sin(2*np.pi*f1*t + ph)
x2 = A2 * np.sin(2*np.pi*f2*t + ph)
x3 = A3 * np.sin(2*np.pi*f3*t + ph)

plt.plot(t,x1)
plt.plot(t,x2)
plt.plot(t,x3)


plt.xlabel('Time'), plt.ylabel('Amplitude')
plt.title('A=[1, 3, 5] V, F= [2, 4, 6] Hz, phi ={}'.format(ph))


