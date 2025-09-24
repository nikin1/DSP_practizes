import numpy as np
import matplotlib.pyplot as plt

#4 Сложение гармонических колебаний c различными частотами
# и начальными фазами (формула 5)
t = np.linspace(0, 2.5, 250)
f = 2
ph = -np.pi / 2

x = 0
for n in range(0, 6):
    x_i = 4 / ((2 * n - 1)* np.pi) * np.cos(2*np.pi*(2*n-1)*f*t + ph)
    x = x + x_i
    plt.plot(t, x_i)

plt.plot(t, x)
plt.title('#4 Сложение гармонических колебаний c различными частотами и начальными фазами (формула 5)')
plt.xlabel('Time (t)')
plt.ylabel('Amplitude (V)')
