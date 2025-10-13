import numpy as np
import matplotlib.pyplot as plt

# Параметры сигнала и среды
fc = 2e9  # частота сигнала, Гц
c = 3e8   # скорость света, м/с
F = 16
V = 10    # скорость абонента, м/с
D = 1000  # расстояние до базовой станции, м
N = 1000  # количество точек траектории
NFFT = 128  # размер ДПФ

# Длина волны и шаг
wavelength = c / fc
delta_x = wavelength / F
ts = delta_x / V  # интервал дискретизации по времени

# Вектор положения абонента (линейный по оси X, 0..100 м)
x = np.arange(N) * delta_x
t = np.arange(N) * ts

# Модель 1: сигнал от базовой станции без рассеивателей
d = D - x  # расстояния BS-MS
k = 2 * np.pi / wavelength
r = np.exp(-1j * k * d)
amplitude = np.abs(r)
phase = np.angle(r)
fs = 1 / ts
f = (np.arange(-NFFT // 2, NFFT // 2) * fs) / N
fd = V / wavelength

# Вывод результатов модели без рассеивателей
print("=== Модель 1: Прямой сигнал от BS ===")
print("Амплитуда первых 10 точек:", amplitude[:10])
print("Фаза первых 10 точек:", phase[:10])
print("Смещение Доплера (Гц):", fd)
print("Ось частот для ДПФ (первые 10):", f[:10])


# Модель 2: сигнал с множеством источников переотражений

# Координаты БС
BSx, BSy = 1000, 1000

num_scatterers = 12  # теперь 12 рассеивателей
radius_sc = 50
angles = 2 * np.pi * np.random.rand(num_scatterers)
radii = radius_sc * np.sqrt(np.random.rand(num_scatterers))
SCx = radii * np.cos(angles)
SCy = radii * np.sin(angles)

# Трек перемещения абонента (MS) вдоль оси X
MSx = np.linspace(0, 100, N)
MSy = np.zeros(N)

# Расстояния BS-SC для всех рассеивателей
dBSSC = np.sqrt((BSx - SCx)**2 + (BSy - SCy)**2)

# Расстояния SC-MS для каждой точки трека и каждого рассеивателя
dSCMS = np.sqrt((SCx[:, None] - MSx)**2 + (SCy[:, None] - MSy)**2)

# Суммарные расстояния
d_total = dBSSC[:, None] + dSCMS

# Комплексная амплитуда суммарного сигнала
r_multi = np.sum(np.exp(-1j * k * d_total), axis=0)
amplitude_multi = np.abs(r_multi)
phase_multi = np.angle(r_multi)

# Спектр Доплера для сигнала с переотражениями
spectrum = np.fft.fftshift(np.fft.fft(r_multi, n=NFFT))
freq_axis = np.fft.fftshift(np.fft.fftfreq(NFFT, d=ts))
power_spectrum = np.abs(spectrum)**2

# Гистограмма модуля сигнала (нормированная)
hist, bin_edges = np.histogram(amplitude_multi, bins=50, density=True)

# Функция распределения (CDF)
cdf = np.cumsum(hist) * np.diff(bin_edges)[0]

# Автокорреляция сигнала
autocorr = np.correlate(r_multi, r_multi, mode='full') / N
lags = np.arange(-N+1, N)

# fs = 120
# ts = 1 / fs
# t = np.ara(8000) * t
# Xc = x1 + j*x2
#

# butterwort () n = 4, wp = 0.1 ws = 0.4 , R = 40, t = [0; 100]
#
#
#









# Визуализация
plt.figure(figsize=(14, 11))

plt.subplot(321)
plt.scatter(SCx, SCy, c='r', label='Рассеиватели')
plt.plot(MSx, MSy, label='Трек абонента (MS)')
plt.scatter([BSx], [BSy], c='b', marker='x', label='Базовая станция (BS)')
plt.title('Сценарий: Рассеиватели, БС и движение MS')
plt.legend()
plt.grid()

plt.subplot(322)
plt.plot(t, amplitude_multi)
plt.title('Модуль принятого сигнала (множество переотражений)')
plt.xlabel('Время, с')
plt.grid()

plt.subplot(323)
plt.plot(t, phase_multi)
plt.title('Фаза принятого сигнала (множество переотражений)')
plt.xlabel('Время, с')
plt.grid()

plt.subplot(324)
plt.plot(freq_axis, power_spectrum)
plt.title('Спектр Доплера')
plt.xlabel('Частота, Гц')
plt.grid()

plt.subplot(325)
plt.plot(bin_edges[:-1], hist)
plt.title('Гистограмма модуля сигнала (нормированная)')
plt.xlabel('Амплитуда')
plt.grid()

plt.subplot(326)
plt.plot(lags, np.abs(autocorr))
plt.title('Автокорреляция принятого сигнала')
plt.xlabel('Сдвиг')
plt.grid()

plt.tight_layout()
plt.show()












plt.figure()
plt.scatter(SCx, SCy, c='r', label='Рассеиватели')
plt.plot(MSx, MSy, label='Трек абонента')
plt.scatter([BSx], [BSy], c='b', marker='x', label='БС')
plt.title('Сценарий: Рассеиватели, Трек, БС')
plt.legend()
plt.grid()

plt.figure()
plt.plot(t, amplitude_multi)
plt.title('Амплитуда принятого сигнала')
plt.xlabel('Время, с')
plt.grid()

plt.figure()
plt.plot(t, phase_multi)
plt.title('Фаза принятого сигнала')
plt.xlabel('Время, с')
plt.grid()

plt.figure()
plt.plot(freq_axis, power_spectrum)
plt.title('Спектр Доплера')
plt.xlabel('Частота, Гц')
plt.grid()

plt.figure()
plt.plot(bin_edges[:-1], hist)
plt.title('Гистограмма модуля сигнала')
plt.xlabel('Амплитуда')
plt.grid()

plt.figure()
plt.plot(lags, np.abs(autocorr))
plt.title('Автокорреляция сигнала')
plt.xlabel('Сдвиг')
plt.grid()

plt.show()

