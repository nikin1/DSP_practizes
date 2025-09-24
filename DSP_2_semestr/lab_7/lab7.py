import numpy as np
import matplotlib.pyplot as plt


d = [0, 1, 1, 0, 1, 1]

sigma = 1 ** 0.5
mean = 0
num_samples = 48 # кол-во



def create_BPSK(d):
    s = []
    
    
    return s



def create_symbol(d):
    t = []
    for i in range(len(d)):
        # print(d[i])
        if d[i] == 0:
            # print("<<0>>")
            for j in range(8):
                t.append(-1)
        else:
            for j in range(8):
                t.append(1)
    return t

def noise_gaus_arr(d, sigma):
    noise_arr = []
    for i in range(len(d)):
        if d[i] == 0:
            noise_arr.append(np.random.normal(-1, sigma, 8))            
        else:
            noise_arr.append(np.random.normal(1, sigma, 8))

    noise = np.concatenate(noise_arr)
    return noise

def graphics(y):

    # Создание массива индексов для оси x
    x = np.arange(len(y))    
    # Отрисовка графика
    plt.figure(figsize=(10, 5))
    # plt.plot(x, t, marker='o', linestyle='-', color='b')  # Используем точки и линию
    plt.plot(x, y, color='b')  # Используем точки и линию

    plt.title('Plot of Array t')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.grid()
    plt.axhline(0, color='black', linewidth=0.5, linestyle='--')  # Добавляем горизонтальную линию на уровне 0
    plt.xticks(x)  # Устанавливаем метки по оси x
    plt.show()


def graphics_2(y, y2):

    # Создание массива индексов для оси x
    x = np.arange(len(y))
    x2 = np.arange(len(y2))
    # Отрисовка графика
    plt.figure(figsize=(10, 5))
    # plt.plot(x, t, marker='o', linestyle='-', color='b')  # Используем точки и линию
    plt.plot(x, y, color='b')  # Используем точки и линию
    plt.plot(x2, y2, color='r')  # Используем точки и линию

    plt.title('Plot of Array t')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.grid()
    plt.axhline(0, color='black', linewidth=0.5, linestyle='--')  # Добавляем горизонтальную линию на уровне 0
    plt.xticks(x)  # Устанавливаем метки по оси x
    plt.show()


def corr_func(Rx):
    S1 = 1
    corr_arr = []
    index = 0
    for i in range(0, len(Rx), 8):
        symbol = Rx[i:i+8]
        # print(symbol)
        summ = 0
        for j in range(len(symbol)):
            
            corr_i = symbol[j] * S1
            summ += corr_i
            corr_arr.append(summ)
            index += 1

    return corr_arr
t = create_symbol(d)
# print(t)

graphics(t)


gaussian_noise = noise_gaus_arr(d, sigma)


print(gaussian_noise)
graphics(gaussian_noise)
graphics_2(t, gaussian_noise)

Rx = gaussian_noise
corr_func(Rx)



# gaussian_noise = np.random.normal(mean, sigma, num_samples)


# plt.figure(figsize=(10, 5))
# plt.hist(gaussian_noise, bins=30, density=True, alpha=0.6, color='g') 



lymbda = 0
corr_arr = corr_func(Rx)


graphics(corr_arr)




# Построить з-сть сигнал, шум
# 1. генерация бит в = [01111]
# 2. s = [-1 1] - символы
# 3. вектор r = s+ n  (сигнал + шум)
# 4. Считаем расстояние м/у полученным сигналом и опорным колебаниям










