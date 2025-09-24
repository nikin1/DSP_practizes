import numpy as np
import matplotlib.pyplot as plt

# Параметры импульса
start = 20
end = 40
length = 60

# Генерация исходного импульса
x = np.arange(length)
pulse = np.zeros(length)
pulse[start:end] = 1

# Добавление шума (амплитуда шума = 200% от амплитуды импульса)
noise = 2 * np.random.randn(length)  # Гауссовский шум
noisy_pulse = pulse + noise

# Визуализация
plt.figure(figsize=(10, 5))
plt.plot(x, pulse, 'b', drawstyle='steps-post', label='Исходный импульс')
plt.plot(x, noisy_pulse, 'r', drawstyle='steps-post', alpha=0.7, label='С шумом')
plt.title('Воздействие сильного шума на прямоугольный импульс')
plt.xlabel('Время')
plt.ylabel('Амплитуда')
plt.ylim(-3, 4)
plt.grid(True)
plt.legend()
plt.show()
