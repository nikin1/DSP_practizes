import numpy as np
import matplotlib.pyplot as plt


def isNaN(num):
    return num != num


f_s = 400
t_s = 1 / f_s
t = np.linspace(0, 2 / 60, 4000)
t_n = np.arange(0, 1000, 1 / f_s)


t_1 = np.arange(-10, 10, t_s)
t_2 = np.linspace(-10, 10, 100)


h = np.sin(np.pi * t_2) /( np.pi * t_2)
for i in range(0, len(h)):
    if isNaN(h[i]):
        print("NAN")
        h[i] = 1

# print(h)



x_1 = np.cos(2 * np.pi * 60 * t)
x_2 = np.cos(2 * np.pi * 340 * t)
x_3 = np.cos(2 * np.pi * 460 * t)

x_1_d = np.cos(2 * np.pi * 60 * t_n)
x_2_d = np.cos(2 * np.pi * 340 * t_n)
x_3_d = np.cos(2 * np.pi * 460 * t_n)




x_1_output = np.convolve(x_1_d, h)





# Вывод 1-ого графика
fig, (ax1, ax2, ax3) = plt.subplots(3)
# fig.suptitle('5 символов')
ax1.stem(t_n, x_1_d)
ax1.plot(t, x_1)

ax2.stem(t_n, x_2_d)
ax2.plot(t, x_2)

ax3.stem(t_n, x_3_d)
ax3.plot(t, x_3)

fig.show()
plt.show()

# Вывод 2-ого графика
plt.plot(t_2, h)
plt.show()



# Output -- Д.З. восстановить косинус
plt.plot(t_n, x_1_output)
plt.show()

