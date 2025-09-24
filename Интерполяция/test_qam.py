#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 17:47:36 2025

@author: santonet
"""

import numpy as np
import matplotlib.pyplot as plt

def qam16_constellation():
    # Создаем идеальные точки 16-QAM (нормированные)
    # По оси I и Q значения из множества {-3, -1, 1, 3}
    # Нормируем мощность до 1
    points = np.array([-3, -1, 1, 3])
    constellation = np.array([x + 1j*y for x in points for y in points])
    constellation /= np.sqrt((np.mean(np.abs(constellation)**2)))

    return constellation

def add_awgn_noise(signal, snr_db):
    # Добавляем белый гауссовский шум к сигналу
    snr_linear = 10**(snr_db / 10)
    power_signal = np.mean(np.abs(signal)**2)
    noise_power = power_signal / snr_linear
    noise = np.sqrt(noise_power/2) * (np.random.randn(len(signal)) + 1j*np.random.randn(len(signal)))
    return signal + noise

# Параметры
num_symbols = 5000  # количество символов
snr_db = 15  # уровень сигнала к шуму в дБ

# Идеальное созвездие 16-QAM
constellation = qam16_constellation()

# Генерируем случайные символы из 16-QAM
data = np.random.randint(0, 16, num_symbols)
tx_symbols = constellation[data]

# Добавляем шум
rx_symbols = add_awgn_noise(tx_symbols, snr_db)

# Визуализация
plt.figure(figsize=(8,8))
plt.plot(rx_symbols.real, rx_symbols.imag, '.', markersize=2, alpha=0.3, label='Приемные символы с шумом')
plt.plot(constellation.real, constellation.imag, 'ro', markersize=8, label='Идеальные символы')
plt.grid(True)
plt.title(f'Созвездие 16-QAM с шумом (SNR = {snr_db} дБ)')
plt.xlabel('In-phase (I)')
plt.ylabel('Quadrature (Q)')
plt.axis('equal')
plt.legend()
plt.show()
