import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# Параметры
fs = 120  # Частота дискретизации, Гц
ts = 1 / fs  # Интервал дискретизации
n = 4  # Порядок фильтра Butterworth
wp = 0.1  # Частота среза фильтра (нормализованная, от 0 до 1)
ws = 0.4  # Заданная, но не используемая напрямую частота стоп-полосы
R = 40  # Параметр, возможно, затухание в дБ, не используется напрямую

# Временной вектор от 0 до 100 секунд с дискретизацией ts
t = np.arange(0, 100, ts)

# Создаем сигнал с распределением Рылеева с параметром масштаба 1
scale = 1.0
rayleigh_signal = np.random.rayleigh(scale, len(t))

# Проектируем фильтр Butterworth низких частот
b, a = butter(n, wp, btype='low', fs=fs)

# Пропускаем сигнал через фильтр (функция filtfilt - для фильтрации с нулевой фазой)
filtered_signal = filtfilt(b, a, rayleigh_signal)

# Строим графики исходного и отфильтрованного сигналов
plt.figure(figsize=(12, 6))
plt.plot(t, rayleigh_signal, label='Original Rayleigh Signal')
plt.plot(t, filtered_signal, label='Filtered Signal (Butterworth)')
plt.title('Rayleigh Distributed Signal filtered by Butterworth Filter')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.show()
