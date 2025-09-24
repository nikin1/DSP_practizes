import numpy as np
import matplotlib.pyplot as plt

# Параметры
A = 3  # амплитуда
f = 5  # частота в Гц
T = 1 / f  # период

# Функция колебания
def x(t, phi):
    return A * np.cos(2 * np.pi * f * t + phi)

# Вычисление коэффициентов a_n и b_n
def compute_fourier_coefficients(n_max, phi):
    a_n = np.zeros(n_max + 1)
    b_n = np.zeros(n_max + 1)

    # a_0
    a_n[0] = (1 / T) * np.trapezoid(x(np.linspace(0, T, 1000), phi), np.linspace(0, T, 1000))

    # a_n и b_n для n = 1, 2, ..., n_max
    for n in range(1, n_max + 1):
        a_n[n] = (2 / T) * np.trapezoid(
            x(np.linspace(0, T, 1000), phi) * np.cos(2 * np.pi * n * np.linspace(0, T, 1000)),
            np.linspace(0, T, 1000))
        b_n[n] = (2 / T) * np.trapezoid(
            x(np.linspace(0, T, 1000), phi) * np.sin(2 * np.pi * n * np.linspace(0, T, 1000)),
            np.linspace(0, T, 1000))

    return a_n, b_n

# Вычисление амплитуд и фаз
def compute_amplitude_phase(a_n, b_n):
    A_n = np.sqrt(a_n**2 + b_n**2)
    phi_n = np.arctan2(b_n, a_n)  # atan2 для правильного определения угла
    return A_n, phi_n

# Построение графиков
def plot_fourier(A_n, phi_n, title_suffix=""):
    n_values = np.arange(len(A_n))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # График A_n
    ax1.stem(n_values[1:], A_n[1:], basefmt=" ")
    ax1.set_title(f'Амплитуды $A_n$ {title_suffix}')
    ax1.set_xlabel('n')
    ax1.set_ylabel('$A_n$')
    ax1.grid()

    # График φ_n
    ax2.stem(n_values[1:], phi_n[1:], basefmt=" ")
    ax2.set_title(r'Фазы $\phi_n$ {title_suffix}')
    ax2.set_xlabel('n')
    ax2.set_ylabel(r'$\phi_n$ (радианы)')
    ax2.grid()

    plt.tight_layout()
    plt.show()

# Основная программа
n_max = 4
phi_1 = 0  # начальная фаза
phi_2 = np.pi / 2  # новая фаза

# Вычисления для начальной фазы φ_1 = 0
a_n_1, b_n_1 = compute_fourier_coefficients(n_max, phi_1)
A_n_1, phi_n_1 = compute_amplitude_phase(a_n_1, b_n_1)

# Графики для φ_1 = 0
plot_fourier(A_n_1, phi_n_1, title_suffix="(φ = 0)")

# Вычисления для новой фазы φ_2 = π/2
a_n_2, b_n_2 = compute_fourier_coefficients(n_max, phi_2)
A_n_2, phi_n_2 = compute_amplitude_phase(a_n_2, b_n_2)

# Графики для φ_2 = π/2
plot_fourier(A_n_2, phi_n_2, title_suffix="(φ = π/2)")
