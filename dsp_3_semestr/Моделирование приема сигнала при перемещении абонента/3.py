import numpy as np
import matplotlib.pyplot as plt

# Параметры задачи
fc = 2e9            # Гц, частота сигнала
c = 3e8             # м/с, скорость света
F = 16              # шаг в длинах волн
V = 10              # м/с, скорость абонента
N = 100             # количество точек траектории
D = 1000            # м, расстояние до BS
NFFT = 128          # размер ДПФ

# Расчеты параметров
lambda_ = c / fc    # длина волны
delta_x = F * lambda_   # шаг по расстоянию
ts = delta_x / V     # шаг по времени
k = 2 * np.pi / lambda_  # волновое число

# Траектория абонента вдоль оси X
x = np.arange(N) * delta_x
t = x / V  # моменты времени

# Расстояния от BS до MS
d = D - x

# Комплексная огибающая принимаемого сигнала
r = np.exp(-1j * k * d)

# Амплитуда и фаза
amplitude = np.abs(r)
phase = np.angle(r)

# Частотная ось для ДПФ
fs = 1 / ts
freq = (np.arange(-NFFT//2, NFFT//2) * fs) / N

# Доплеровское смещение через спектр
r_padded = np.zeros(NFFT, dtype=complex)
r_padded[:N] = r
R_fft = np.fft.fftshift(np.fft.fft(r_padded))
doppler_freq_index = np.argmax(np.abs(R_fft))
doppler_frequency = freq[doppler_freq_index]

# Результаты
print("Доплеровское смещение (Гц):", doppler_frequency)

# Графики амплитуды и фазы
plt.figure(figsize=(12,5))
plt.subplot(121)
plt.plot(t, amplitude)
plt.title("Амплитуда сигнала по времени")
plt.xlabel("Время (с)")
plt.grid()

plt.subplot(122)
plt.plot(t, phase)
plt.title("Фаза сигнала по времени")
plt.xlabel("Время (с)")
plt.grid()
plt.show()
