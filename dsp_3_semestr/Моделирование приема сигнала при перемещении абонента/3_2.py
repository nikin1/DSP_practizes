import numpy as np
import matplotlib.pyplot as plt

# Параметры модели
fc = 2e9            # центральная частота 2 ГГц
c = 3e8             # скорость света, м/с
F = 16              # шаг по длинам волн
V = 10              # скорость, м/с
N = 20              # количество точек траектории (для экономии)
lambda_ = c / fc
delta_x = F * lambda_
ts = delta_x / V
k = 2 * np.pi / lambda_

# Частотная сетка
f_start = 1.9975e9  # 1997.5 МГц
f_stop = 2.0025e9   # 2002.5 МГц
freq_step = 1e4     # 0.01 МГц
freq_grid = np.arange(f_start, f_stop, freq_step)
Nf = len(freq_grid)

# Траектория движения MS
x = np.arange(N) * delta_x
t = x / V

# Координаты рассеивателей (2 источника)
SCx = np.array([300, 600])   # м
# eSCy = np.array([0, 0])
SCy = np.array([200, -200])  # или большие значения, например ±100 м

alpha = np.array([1.0, 0.5]) # коэффициенты передачи

# Координаты BS
BSx, BSy = 1000, 0

# Расчет расстояний BS-SC
d_BS_SC = np.sqrt((BSx - SCx)**2 + (BSy - SCy)**2)

# Результирующая частотная характеристика канала:
# размерность: N (по времени) x Nf (по частоте)
H = np.zeros((N, Nf), dtype=complex)

for n in range(N):
    MSx = x[n]
    MSy = 0
    d_SC_MS = np.sqrt((SCx - MSx)**2 + (SCy - MSy)**2)
    d_total = d_BS_SC + d_SC_MS  # суммарный путь для каждого рассеивателя

    for m, f in enumerate(freq_grid):
        lambda_f = c / f
        k_f = 2 * np.pi / lambda_f
        H[n, m] = np.sum(alpha * np.exp(-1j * k_f * d_total))

# Визуализация исходного расположения
plt.figure(figsize=(8,6))
plt.scatter([BSx], [BSy], c='green', marker='*', s=200, label='БС')
plt.scatter(SCx, SCy, c='red', label='Рассеивающие объекты')
plt.plot(x, np.zeros_like(x), 'b-', label='Траектория MS')
plt.title('Расположение БС, рассеивателей и траектории абонента')
plt.xlabel('x, м')
plt.ylabel('y, м')
plt.legend()
plt.grid()
plt.show()

# Модуль частотной характеристики по времени по центральной частоте
center_idx = Nf // 2
H_module_center = np.abs(H[:, center_idx])

# График модуля ЧХ по времени и частоте (водопад)
plt.figure(figsize=(10,6))
plt.imshow(20*np.log10(np.abs(H.T)), aspect='auto', extent=[t[0], t[-1], freq_grid[0]/1e9, freq_grid[-1]/1e9], origin='lower')
plt.colorbar(label='Модуль ЧХ, дБ')
plt.xlabel('Время, с')
plt.ylabel('Частота, ГГц')
plt.title('Временно-частотная характеристика канала')
plt.show()


from mpl_toolkits.mplot3d import Axes3D  # нужен для 3D-построений

# Сетка по времени (N точек)
T, F = np.meshgrid(t, freq_grid / 1e9)  # частоты в ГГц для удобства

# Модуль комплексной частотной характеристики (в дБ)
Z = 20 * np.log10(np.abs(H).T)  # транспонируем H, чтобы размерности совпали

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Построение поверхности
surf = ax.plot_surface(T, F, Z, cmap='viridis', edgecolor='none')

ax.set_xlabel('Время, с')
ax.set_ylabel('Частота, ГГц')
ax.set_zlabel('Модуль ЧХ, дБ')
ax.set_title('3D временно-частотная характеристика канала')
ax.view_init(elev=30, azim=45)

# ax.contour3D(T, F, Z, 50, cmap='viridis')
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Амплитуда, дБ')
plt.show()

print(f'Z min: {Z.min()}, Z max: {Z.max()}')
print(f'Частотный диапазон: {freq_grid[0]/1e9} ГГц - {freq_grid[-1]/1e9} ГГц c шагом {freq_grid[1]-freq_grid[0]} Гц')







# Импульсная характеристика с помощью обратного FFT по частотам для каждой точки времени
Ht = np.fft.ifft(H, axis=1)
tau_max = 1 / freq_step
Ntau = Nf
tau = np.arange(Ntau) * (tau_max / Ntau)

# График импульсной характеристики в выбранной точке траектории (например, по середине)
mid_idx = N // 2
plt.figure(figsize=(8,4))
plt.plot(tau * 1e6, np.abs(Ht[mid_idx, :]))
plt.title('Импульсная характеристика канала (точка по середине траектории)')
plt.xlabel('Время задержки, мкс')
plt.ylabel('Амплитуда')
plt.grid()
plt.show()

# График модуля ЧХ во всей полосе в середине траектории
plt.figure(figsize=(8,4))
plt.plot((freq_grid-1.999e9)*1e-6, 20*np.log10(np.abs(H[mid_idx, :])))
plt.title('Модуль частотной характеристики (середина траектории)')
plt.xlabel('Отклонение частоты, МГц')
plt.ylabel('Амплитуда, дБ')
plt.grid()
plt.show()

# График фазы ЧХ во всей полосе в середине траектории
plt.figure(figsize=(8,4))
plt.plot((freq_grid-1.999e9)*1e-6, np.angle(H[mid_idx, :]))
plt.title('Фаза частотной характеристики (середина траектории)')
plt.xlabel('Отклонение частоты, МГц')
plt.ylabel('Фаза, рад')
plt.grid()
plt.show()
