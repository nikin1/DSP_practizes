import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft

def generator_sin(f, fs):
    t_total =  1

    t = np.arange(0, t_total, 1/fs)
    y = np.sin(2 * np.pi * f * t)

    return y, t

def DPF(x, N):
    X = fft(x, N) / N
    
    plt.figure
    k = np.arange(0, N)
    plt.stem(k, abs(X))
    plt.xlabel("Частота (Гц)")
    plt.ylabel("Амплитуда")

def visual_graph(t, y, stem_flag):

    plt.figure(figsize=(10, 4))
    if stem_flag:
        plt.stem(t, y)
    plt.plot(t, y, 'b-', alpha=0.5)
    plt.title('Дискретизация')
    plt.xlabel('Время (сек)')
    plt.ylabel('Амплитуда')
    plt.grid(True)
    plt.legend()
    plt.show()

f = 10
fs = 100

y, t = generator_sin(f, fs)

DPF(y, fs)

visual_graph(t, y, 0)