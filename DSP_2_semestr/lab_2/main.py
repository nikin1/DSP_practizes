import numpy as np
import matplotlib.pyplot as plt

# Параметры
m = 0.55  # Среднее
sigma = 3  # Стандартное отклонение
N = 1000  # Количество случайных величин
alpha = 0.8  # Параметр автокорреляции для AR(1)

# Генерация коррелированного нормального СП (AR(1) процесс)
realization = np.zeros(N)
realization[0] = np.random.normal(m, sigma)  # Первое значение

for t in range(1, N):
    realization[t] = m + alpha * (realization[t-1] - m) + np.random.normal(0, sigma)

# Интервалы между сечениями
time_intervals = [0, 3, 5, 7]
average_products = []

for interval in time_intervals:
    products = []
    if interval < N:  # Проверка, чтобы избежать выхода за пределы массива
        value_t1 = realization[0]  # Первое значение по индексу 0
        value_t2 = realization[interval]  # Второе значение с учетом интервала
        products.append(value_t1 * value_t2)
    
    average_product = np.mean(products)
    average_products.append(average_product)

# Вывод средних произведений
for interval, avg in zip(time_intervals, average_products):
    print(f'Среднее произведение для интервала {interval}: {avg}')

# Вычисление АКФ для одной реализации
tau_values = np.arange(0, N//2, 1)
acf_values = []

for tau in tau_values:
    if tau < len(realization):  # Проверка, чтобы избежать выхода за пределы массива
        if len(realization[:-tau]) > 0 and len(realization[tau:]) > 0:  # Проверка на пустоту массивов
            acf_sum = realization[:-tau].dot(realization[tau:])  # Произведение значений
            acf_values.append(acf_sum / (N - tau))  # Среднее значение
        else:
            acf_values.append(0)  # Если массив пуст
    else:
        acf_values.append(0)  # Если tau слишком велико

# Построение графика АКФ
plt.figure(figsize=(12, 6))
plt.plot(tau_values, acf_values, label='АКФ', color='purple')
plt.title('Автокорреляционная функция (по одной реализации)')
plt.xlabel('τ (лаг)')
plt.ylabel('АКФ')
plt.legend()
plt.grid()
plt.show()

# Определение интервала корреляции
correlation_threshold = 0.1  # Порог для определения интервала корреляции
correlation_interval = np.where(np.array(acf_values) < correlation_threshold)[0]
if correlation_interval.size > 0:
    print(f'Интервал корреляции: от 0 до {correlation_interval[0]}')
else:
    print('Интервал корреляции не найден.')
