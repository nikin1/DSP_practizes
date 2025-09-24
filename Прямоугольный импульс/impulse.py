import numpy as np
import matplotlib.pyplot as plt

# Параметры прямоугольного импульса
start = 20
end = 40
length = 60

# Создание массива времени
x = np.arange(0, length)

# Создание прямоугольного импульса
pulse = np.zeros(length)
pulse[start:end] = 1

# Визуализация
plt.figure(figsize=(8, 4))
plt.plot(x, pulse, drawstyle='steps-post')
plt.title('Прямоугольный импульс')
plt.xlabel('Время')
plt.ylabel('Амплитуда')
plt.ylim(-0.1, 1.1)
plt.grid(True)
plt.show()
