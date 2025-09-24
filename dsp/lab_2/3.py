import numpy as np
import matplotlib.pyplot as plt


#3 Сложение гармонических колебаний c различ фазами
t = np.linspace(0, 2.5, 250)
A1 = 6
A2 = 6
A3 = 6
f = 2
ph1 = -30
ph2 = -60
ph3 = 90

x1 = A1 * np.cos(2*np.pi*f*t + ph1)
x2 = A2 * np.cos(2*np.pi*f*t + ph2)
x3 = A3 * np.cos(2*np.pi*f*t + ph3)
x = x1 + x2 + x3
plt.plot(t, x1)
plt.plot(t, x2)
plt.plot(t, x3)
plt.plot(t, x)
plt.title('#3 Сложение гармонических колебаний c различными фазами')

plt.xlabel('Time (t)')
plt.ylabel('Amplitude (V)')

