import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

fs = 10000  # Частота дискретизации (10 кГц)
duration = 1  # Длительность сигнала (в секундах)
N = fs * duration  # Общее количество точек
num_realizations = 1000  # Количество реализаций для усреднения

# Параметры фильтров
fc_low = 3000  # Частота среза ФНЧ (3 кГц)
f1_band = 1000  # Первая частота среза ПФ (1 кГц)
f2_band = 4000  # Вторая частота среза ПФ (4 кГц)

def generate_white_noise(N):
    return np.random.normal(0, 1, N)

def apply_filters(white_noise):
    # Нормализация частот среза
    normalized_fc_low = fc_low / (fs / 2)  # Нормализованная частота среза ФНЧ
    normalized_f1_band = f1_band / (fs / 2)  # Нормализованная первая частота среза ПФ
    normalized_f2_band = f2_band / (fs / 2)  # Нормализованная вторая частота среза ПФ

    # Проверка нормализованных частот
    if not (0 < normalized_fc_low < 1):
        raise ValueError(f"Нормализованная частота среза ФНЧ {normalized_fc_low} должна быть в диапазоне (0, 1)")
    if not (0 < normalized_f1_band < 1 and 0 < normalized_f2_band < 1):
        raise ValueError(f"Нормализованные частоты среза ПФ {normalized_f1_band}, {normalized_f2_band} должны быть в диапазоне (0, 1)")

    # ФНЧ
    b_low, a_low = scipy.signal.ellip(6, 0.5, 50, normalized_fc_low, btype='low')
    low_passed_signal = scipy.signal.lfilter(b_low, a_low, white_noise)

    # ПФ
    b_band, a_band = scipy.signal.ellip(6, 0.5, 50, [normalized_f1_band, normalized_f2_band], btype='band')
    band_passed_signal = scipy.signal.lfilter(b_band, a_band, white_noise)

    return low_passed_signal, band_passed_signal

def autocorrelation(signal):
    n = len(signal)
    mean = np.mean(signal)
    c = np.correlate(signal - mean, signal - mean, mode='full')
    return c[n - 1:]

def power_spectral_density(signal):
    f, Pxx = scipy.signal.welch(signal, fs, nperseg=1024)
    return f, Pxx

low_passed_signals = []
band_passed_signals = []
for i in range(num_realizations):
    white_noise = generate_white_noise(N)
    low_passed_signal, band_passed_signal = apply_filters(white_noise)
    low_passed_signals.append(low_passed_signal)
    band_passed_signals.append(band_passed_signal)

# Далее можно продолжить с анализом и визуализацией выходных сигналов.


# Упрощение АКФ и СПМ
mean_low_passed_signal = np.mean(low_passed_signals, axis=0)
mean_band_passed_signal = np.mean(band_passed_signals, axis=0)

# Вычислениеe АКФ и СПМ
autocorr_white_noise = autocorrelation(generate_white_noise(N))
autocorr_low_passed = autocorrelation(mean_low_passed_signal)
autocorr_band_passed = autocorrelation(mean_band_passed_signal)


f_low_passed, psd_low_passed = power_spectral_density(mean_low_passed_signal)
f_band_passed, psd_band_passed = power_spectral_density(mean_band_passed_signal)


# plt.figure(figsize=(15, 10))

# Входной белый шум
plt.figure()
# plt.subplot(3, 2, 1)
plt.title("Входной белый шум")
plt.plot(generate_white_noise(N)[:1000])  # Отображение первых 1000 точек
plt.xlabel("Время (дискреты)")
plt.ylabel("Амплитуда")
plt.show()

# АКФ входного белого шума
plt.figure()
plt.title("АКФ входного белого шума")
plt.plot(autocorr_white_noise[:1000])  # Отображение первых 1000 лагов
plt.xlabel("Лаг")
plt.ylabel("АКФ")
plt.show()

# Выходной сигнал ФНЧ
plt.figure()
plt.title("Выходной сигнал ФНЧ")
plt.plot(mean_low_passed_signal[:1000])  # Отображение первых 1000 точек
plt.xlabel("Время (дискреты)")
plt.ylabel("Амплитуда")
plt.show()

# АКФ выходного сигнала ФНЧ
plt.figure()
plt.title("АКФ выходного сигнала ФНЧ")
plt.plot(autocorr_low_passed[:1000])  # Отображение первых 1000 лагов
plt.xlabel("Лаг")
plt.ylabel("АКФ")
plt.show()

# Выходной сигнал ПФ
plt.figure()
plt.title("Выходной сигнал ПФ")
plt.plot(mean_band_passed_signal[:1000])  # Отображение первых 1000 точек
plt.xlabel("Время (дискреты)")
plt.ylabel("Амплитуда")
plt.show()

# СПМ выходного сигнала ПФ
plt.figure()
plt.title("СПМ выходного сигнала ПФ")
plt.semilogy(f_band_passed, psd_band_passed)  # Логарифмическая шкала для СПМ
plt.xlabel("Частота (Гц)")
plt.ylabel("СПМ (дБ/Гц)")

plt.tight_layout()
plt.show()




# # Усреднение АКФ по реализациям для выходных сигналов
# mean_autocorr_low_passed = np.mean([autocorrelation(signal) for signal in low_passed_signals], axis=0)
# mean_autocorr_band_passed = np.mean([autocorrelation(signal) for signal in band_passed_signals], axis=0)

# # Усреднение СПМ по реализациям для выходных сигналов
# mean_psd_low_passed = np.mean([power_spectral_density(signal)[1] for signal in low_passed_signals], axis=0)
# mean_psd_band_passed = np.mean([power_spectral_density(signal)[1] for signal in band_passed_signals], axis=0)

# # Построение графиков усредненных АКФ и СПМ
# plt.figure(figsize=(15, 10))

# # Усредненная АКФ выходного сигнала ФНЧ
# plt.subplot(2, 2, 1)
# plt.title("Усредненная АКФ выходного сигнала ФНЧ")
# plt.plot(mean_autocorr_low_passed[:1000])  # Отображение первых 1000 лагов
# plt.xlabel("Лаг")
# plt.ylabel("АКФ")

# # Усредненная АКФ выходного сигнала ПФ
# plt.subplot(2, 2, 2)
# plt.title("Усредненная АКФ выходного сигнала ПФ")
# plt.plot(mean_autocorr_band_passed[:1000])  # Отображение первых 1000 лагов
# plt.xlabel("Лаг")
# plt.ylabel("АКФ")

# # Усредненная СПМ выходного сигнала ФНЧ
# plt.subplot(2, 2, 3)
# plt.title("Усредненная СПМ выходного сигнала ФНЧ")
# plt.semilogy(f_low_passed, mean_psd_low_passed)  # Логарифмическая шкала для СПМ
# plt.xlabel("Частота (Гц)")
# plt.ylabel("СПМ (дБ/Гц)")

# # Усредненная СПМ выходного сигнала ПФ
# plt.subplot(2, 2, 4)
# plt.title("Усредненная СПМ выходного сигнала ПФ")
# plt.semilogy(f_band_passed, mean_psd_band_passed)  # Логарифмическая шкала для СПМ
# plt.xlabel("Частота (Гц)")
# plt.ylabel("СПМ (дБ/Гц)")

# plt.tight_layout()
# plt.show()
