import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

f = 5
T = 1 / f
Ts = 0.01
#t = np.linspace(− 0.1, 0.1, 100, endpoint=False)
A = 3
ph = 0
t = np.arange(-0.1, 0.1, Ts)
s = A * np.cos(2 * np.pi * f * t + ph)
sc = np.cos(2 * np.pi * f * t)
ss = np.sin(2 * np.pi * f * t)
m1 = s * sc
m2 = s * ss

plt.plot(t, s, t, sc, t, m1)
plt.ylim(-3, 3)

a1 = 1 / T * np.sum(m1) * Ts
print(a1)




