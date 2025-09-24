#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 16:51:04 2025

@author: santonet
"""

import numpy as np
import matplotlib.pyplot as plt
# Определяем функцию плотности вероятности нормального распределения

def W(x, mx, q):
return (1 / np.sqrt(2 * np.pi * q**2)) * np.exp(-((x - mx)2) / (2 * q2))
Создаем массив значений x от -5 до 5 с шагом 0.01

x = np.arange(-5, 5, 0.01)
# Определяем параметры для графиков

params = [
(0, 1), # mx = 0, q = 1
(0, 3), # mx = 0, q = 3
(0, 0.2), # mx = 0, q = 0.2
(-1, 1) # mx = -1, q = 1
]
# Создаем график

plt.figure(figsize=(10, 6))

for mx, q in params:
plt.plot(x, W(x, mx, q), label=f'mx={mx}, q={q}')
Настройка графика

plt.title('Плотность вероятности нормального распределения')
plt.xlabel('x')
plt.ylabel('Плотность вероятности')
plt.axhline(0, color='black', lw=0.5, ls='--')
plt.axvline(0, color='black', lw=0.5, ls='--')
plt.grid()
plt.legend()
plt.show()

t = np.linspace(0, 3, 5000)
for m, q in params:
s1 = np.sqrt(q)
# plt.plot(x, W(x, mx, q), label=f'mx={mx}, q={q}')

# Генерируем выборку случайных значений с нормальным распределением
xn = np.random.normal(m, s1, len(t))

# Создаем график выборки
plt.figure(figsize=(12, 6))

# График выборки
plt.subplot(1, 2, 1)
plt.hist(xn, bins=50, density=True, alpha=0.6, color='g')
plt.title('Гистограмма выборки')
plt.xlabel('Значения')
plt.ylabel('Плотность')

# График плотности распределения
plt.subplot(1, 2, 2)
# Плотность распределения
x = np.linspace(m - 4*s1, m + 4*s1, 1000)  # диапазон для плотности
plt.plot(x, (1 / (np.sqrt(2 * np.pi) * s1)) * np.exp(-0.5 * ((x - m) / s1) ** 2), color='blue')
plt.title('Плотность вероятности нормального распределения')
plt.xlabel('Значения')
plt.ylabel('Плотность')

plt.tight_layout()
plt.show()
