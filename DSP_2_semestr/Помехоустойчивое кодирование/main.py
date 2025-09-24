import numpy as np

# Порождающая матрица G (7,4) кода Хэмминга
G = np.array([
    [1, 0, 0, 0, 1, 1, 0],
    [0, 1, 0, 0, 1, 0, 1],
    [0, 0, 1, 0, 0, 1, 1],
    [0, 0, 0, 1, 1, 1, 1]
])



m = [0, 1, 0, 1]


c = np.mod(m @ G, 2)



# Генерируем все 4-битные комбинации
info_bits = [np.array([int(b) for b in f"{i:04b}"]) for i in range(16)]

# Вычисляем кодовые слова
codewords = [np.mod(bits @ G, 2) for bits in info_bits]

# Выводим таблицу
print("Инф. биты | Кодовое слово")
for bits, cw in zip(info_bits, codewords):
    print(f"{bits} : {cw}")

# Находим H
P = G[:, 4:]
H = np.hstack((P.T, np.eye(3, dtype=int)))
print("\nПроверочная матрица H:")
print(H)


# Создаем все векторы ошибок (одна "1" в каждой позиции)
error_vectors = np.eye(7, dtype=int)

# Вычисляем синдромы
syndromes = np.mod(error_vectors @ H.T, 2)

# Выводим таблицу
print("Вектор ошибки | Синдром | Позиция ошибки")
for i in range(7):
    print(f"{error_vectors[i]} : {syndromes[i]} → {i+1}")















# # Все возможные 4-битные информационные векторы (от 0000 до 1111)
# info_bits = np.array([list(map(int, np.binary_repr(i, width=4))) for i in range(16)])

# # Вычисление кодовых слов: c = u * G (mod 2)
# codewords = np.mod(info_bits @ G, 2)

# # Вывод таблицы [инф. биты : кодовое слово]
# print("Информационные биты | Кодовое слово")
# for u, c in zip(info_bits, codewords):
#     print(f"{u} : {c}")

# # Проверочная матрица H (из условия G @ H.T = 0 mod 2)
# P = G[:, 4:]  # Последние 3 столбца G
# H = np.hstack((P.T, np.eye(3, dtype=int)))  # H = [P^T | I_3]

# print("\nПроверочная матрица H:")
# print(H)



