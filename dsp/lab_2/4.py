import numpy as np
import matplotlib.pyplot as plt

#4 Сложение гармонических колебаний c различными частотами
# и начальными фазами (формула 4)
t = np.linspace(0, 2.5, 250)
A1 = 4 / np.pi
A2 = 4 / (3 * np.pi)
f = 2
ph = -np.pi / 2

x1 = A1 * np.cos(2*np.pi*f*t + ph)
x2 = A2 * np.cos(2*np.pi*3*f*t + ph)
x = x1 + x2
plt.plot(t, x1)
plt.plot(t, x2)
plt.plot(t, x)
plt.title('#4 Сложение гармонических колебаний c различными частотами и начальными фазами (формула 4)')

plt.xlabel('Time (t)')
plt.ylabel('Amplitude (V)')