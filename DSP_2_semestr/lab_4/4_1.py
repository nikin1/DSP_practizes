#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 16:52:22 2025

@author: santonet
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

# Параметры
fs = 10000  # Частота дискретизации (10 кГц)
duration = 1  # Длительность сигнала (в секундах)
N = fs * duration  # Общее количество точек

# Параметры фильтров
fc_low = 3000  # Частота среза ФНЧ (3 кГц)
f1_band = 1000  # Первая частота среза ПФ (1 кГц)
f2_band = 4000  # Вторая частота среза ПФ (4 кГц)

# Генерация белого шума
def generate_white_noise(N):
    return np.random.normal(0, 1, N)

# Применение фильтров
def apply_filters(white_noise):
    # Нормализация частот среза
    normalized_fc_low = fc_low / (fs / 2)
    normalized_f1_band = f1_band / (fs / 2)
    normalized_f2_band = f2_band / (fs / 2)

    # ФНЧ
    b_low, a_low = scipy.signal.ellip(6, 0.5, 50, normalized_fc_low, btype='low')
    low_passed_signal = scipy.signal.lfilter(b_low, a_low, white_noise)

    # ПФ
    b_band, a_band = scipy.signal.ellip(6, 0.5, 50, [normalized_f1_band, normalized_f2_band], btype='band')
    band_passed_signal = scipy.signal.lfilter(b_band, a_band, white_noise)

    return low_passed_signal, band_passed_signal

# Визуализация частотной характеристики
def plot_frequency_response(b, a):
    w, h = scipy.signal.freqz(b, a)
    plt.figure()
    plt.title('Digital filter frequency response')
    plt.plot(w / np.pi * fs / 2, 20 * np.log10(abs(h)), 'b')
    plt.ylabel('Amplitude (dB)', color='b')
    plt.xlabel('Frequency (Hz)')
    plt.grid()
    plt.show()

# Генерация и фильтрация белого шума
white_noise = generate_white_noise(N)
low_passed_signal, band_passed_signal = apply_filters(white_noise)

# Визуализация частотных характеристик
plot_frequency_response(b_low, a_low)  # ФНЧ
plot_frequency_response(b_band, a_band)  # ПФ

