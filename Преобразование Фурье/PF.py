import numpy as np
import matplotlib.pyplot as plt


def plot_amplitude_spectrum(amplitudes):
    """
    Строит график амплитудного спектра гармоник.

    Параметры:
    amplitudes : array_like
        Массив амплитуд гармоник.
    """
    harmonics = np.arange(1, len(amplitudes) + 1)
    plt.figure(figsize=(8, 4))
    plt.stem(harmonics, amplitudes, basefmt=" ")
    plt.title('Амплитудный спектр')
    plt.xlabel('Номер гармоники')
    plt.ylabel('Амплитуда')
    plt.grid(True)
    plt.show()

def plot_phase_spectrum(phases):
    """
    Строит график фазового спектра гармоник.

    Параметры:
    phases : array_like
        Массив фаз гармоник в радианах.
    """
    harmonics = np.arange(1, len(phases) + 1)
    plt.figure(figsize=(8, 4))
    plt.stem(harmonics, phases, basefmt=" ")
    plt.title('Фазовый спектр')
    plt.xlabel('Номер гармоники')
    plt.ylabel('Фаза, рад')
    plt.grid(True)
    plt.show()






















# Параметры синусоидального сигнала
A = 1.0          # амплитуда
f0 = 5.0         # частота сигнала, Гц
phi0 = np.pi/4   # начальная фаза, радианы
T = 1.0          # длительность сигнала, секунды
fs = 1000        # частота дискретизации, Гц

# Временная ось
t = np.linspace(0, T, int(fs*T), endpoint=False)

# Исходный сигнал
signal = A * np.sin(2 * np.pi * f0 * t + phi0)

# Численное вычисление коэффициентов ряда Фурье для первых 5 гармоник
N = 5

a0 = (2/T) * np.trapezoid(signal, t)  # заменено np.trapz на np.trapezoid

a = np.zeros(N)
b = np.zeros(N)

for k in range(1, N+1):
    print("Гармоника №: ", k)
    cos_k = np.cos(2 * np.pi * k * t / T)
    sin_k = np.sin(2 * np.pi * k * t / T)
    a[k-1] = (2/T) * np.trapezoid(signal * cos_k, t)
    b[k-1] = (2/T) * np.trapezoid(signal * sin_k, t)
    print("a and b:", a[k-1], b[k-1])


# Амплитуды и фазы гармоник
amplitudes = np.sqrt(a**2 + b**2)
phases = np.arctan2(-b, a)

# Визуализация
plt.figure(figsize=(12, 8))

plt.subplot(3,1,1)
plt.plot(t, signal, label='Исходный сигнал')
plt.title('Синусоидальное колебание')
plt.xlabel('Время, с')
plt.ylabel('Амплитуда')
plt.grid()
plt.legend()

plt.subplot(3,1,2)
plt.stem(range(1, N+1), amplitudes, basefmt=" ")  # убран use_line_collection
plt.title('Амплитуды гармоник ряда Фурье')
plt.xlabel('Номер гармоники k')
plt.ylabel('Амплитуда')
plt.grid()

plt.subplot(3,1,3)
plt.stem(range(1, N+1), phases, basefmt=" ")  # убран use_line_collection
plt.title('Фазы гармоник ряда Фурье')
plt.xlabel('Номер гармоники k')
plt.ylabel('Фаза, рад')
plt.grid()

plt.tight_layout()
plt.show()



plot_amplitude_spectrum(amplitudes)
plot_phase_spectrum(phases)
