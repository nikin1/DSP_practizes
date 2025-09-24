import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Определяем функцию плотности вероятности нормального распределения
def W(x, mx, q):
    return (1 / np.sqrt(2 * np.pi * q**2)) * np.exp(-((x - mx)**2) / (2 * q**2))

# Создаем массив значений x от -5 до 5 с шагом 0.01
x = np.arange(-5, 5, 0.01)

# Определяем параметры для графиков
params = [
    (0, 1),   # mx = 0, q = 1
    (0, 3),   # mx = 0, q = 3
    (0, 0.2), # mx = 0, q = 0.2
    (-1, 1)   # mx = -1, q = 1
]

# Создаем график
plt.figure(figsize=(10, 6))

for mx, q in params:
    plt.plot(x, W(x, mx, q), label=f'mx={mx}, q={q}')

# Настройка графика
plt.title('Плотность вероятности нормального распределения')
plt.xlabel('x')
plt.ylabel('Плотность вероятности')
plt.axhline(0, color='black', lw=0.5, ls='--')
plt.axvline(0, color='black', lw=0.5, ls='--')
plt.grid()
plt.legend()
plt.show()







# Генерируем временной вектор
t = np.linspace(0, 3, 5000)
N = len(t)
# Создаем график для каждого набора параметров
for m, q in params:
    s1 = np.sqrt(q)
    
    # Генерируем выборку случайных значений с нормальным распределением
    xn = np.random.normal(m, s1, N)
    
    
    
    M = sum(xn) / N
    
    
    q_2 = sum(xn ** 2) / N - M**2
    print(f"Эмпирическое Математическое ожидание (E[X]): {M}")
    print(f"Эмпирическая Дисперсия (Var[X]): {q_2}")
    
    
    W_x = W(xn, M, q_2)
    
    # M_quad = quad(lambda x: xn * W_x, -np.inf, np.inf)
    # Q_quad = quad(lambda x: xn ** 2 * W_x, -np.inf, np.inf) - M_quad
    # print('M_quad: ', M_quad)
    # print('Q_quad: ', Q_quad)
    
    
    # Вычисление математического ожидания (E[X])
    E_X, _ = quad(lambda x: np.mean(xn) * W(x, M, q_2), -np.inf, np.inf)    

    # Вычисление E[X^2]
    # E_X2, _ = quad(lambda x: xn**2 * W_x, -np.inf, np.inf)
    E_X2, _ = quad(lambda x: np.mean(xn ** 2) * W(x, M, q_2), -np.inf, np.inf)
    # Вычисление дисперсии (Var[X])
    Var_X = E_X2 - E_X**2
    
    # Вывод результатов
    print(f"Математическое ожидание по плотности распределения (E[X]): {E_X}")
    print(f"Дисперсия по плотности распределения (Var[X]): {Var_X}")
        
        
    
    
    
    # Шаг для бинов
    bin_width = 0.1  # Можно изменить по желанию
    bins = np.arange(m - 4*s1, m + 4*s1 + bin_width, bin_width)  # Границы бинов
    
    # Подсчет количества попаданий в каждый сегмент
    counts, edges = np.histogram(xn, bins=bins)
    
    # Центральные значения сегментов
    bin_centers = 0.5 * (edges[1:] + edges[:-1])
    
    # Нормируем значения счетчика
    normalized_counts = counts / (len(xn) * bin_width)
    
    # Создаем график выборки
    plt.figure(figsize=(12, 6))

    # График эмпирической плотности распределения
    plt.subplot(1, 2, 1)
    plt.bar(bin_centers, normalized_counts, width=bin_width, alpha=0.6, color='g', align='center')
    plt.title('Эмпирическая плотность распределения')
    plt.xlabel('Значения')
    plt.ylabel('Плотность')

    # График плотности распределения
    plt.subplot(1, 2, 2)
    x = np.linspace(m - 4*s1, m + 4*s1, 1000)  # диапазон для плотности
    plt.plot(x, (1 / (np.sqrt(2 * np.pi) * s1)) * np.exp(-0.5 * ((x - m) / s1) ** 2), color='blue')
    plt.title('Плотность вероятности нормального распределения')
    plt.xlabel('Значения')
    plt.ylabel('Плотность')
    
    plt.tight_layout()
    plt.show()

