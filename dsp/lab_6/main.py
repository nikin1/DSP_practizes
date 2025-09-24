import numpy as np
import matplotlib.pyplot as plt

#print("Hello, world!")
R = 10 ** 5
C = 10 ** -6
tau = R * C
w_1 = 60
A = 1

w = np.linspace(0, 2 / tau * 2 * np.pi, 100)
H_j_w = 1 / np.sqrt(1 + (w * R * C) ** 2)
H_j_w_1 = 1 / np.sqrt(1 + (w_1 * R * C) ** 2)

fi_w = - np.arctan(w * R * C)
fi_w_1 = - np.arctan(w_1 * R * C)

#print(t)

# plt.set_title('Фазы ')
# plt.set_ylabel('(радианы)')
# plt.set_xlabel('n')

#fig, (ax1, ax2, ax3) = plt.subplots(3)
#fig.suptitle('5 символов')
#ax1.plot(w, H_j_w)
#ax2.plot(w, fi_w)


plt.title("АЧХ")
plt.ylabel("|H(jw)|")
plt.xlabel("w")
plt.plot(w, H_j_w, color="green")
plt.show()


plt.title("ФЧХ")
plt.ylabel("fi(w)")
plt.xlabel("w")
plt.plot(w, fi_w, color = "r")
plt.show()

t = np.linspace(0, 1, 100)
S = A * np.cos(w_1 * t)
y = A * H_j_w_1 * np.cos(w_1 * t + fi_w_1)

plt.plot(t, S, t, y)
plt.xlabel("w")
plt.title(f"w1 = {w_1}")
plt.show()


