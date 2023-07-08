import scipy
from numpy import *

# Входные данные:
# X - Список из 5000 изображений цифр 20х20px представленных в виде списка из 400 элементов (матрица 5000 х 400)
from displayData import displayData
from predict import predict

X = scipy.io.loadmat("test_set.mat")['X']
# Y - Список из 5000 чисел представлеенных на изображениях выше (Вектор 5000 х 1)
Y = scipy.io.loadmat("test_set.mat")['y']

# Веса первого и второго слоя
Theta1 = scipy.io.loadmat("weights.mat")["Theta1"]
Theta2 = scipy.io.loadmat("weights.mat")["Theta2"]


m = len(X)

# 100 случайных изображений из массива X
# displayData(array([X[i] for i in [random.permutation(m)[i] for i in range(0, 100)]]), "Sample")


# Вычисление эффективности предсказания
predicted = predict(X, Theta1, Theta2)
guessed = ones(m)
for i in range(0, m):
    guessed[i] = int(Y[i][0] == predicted[i])
print(f"Efficiency of predicting: {(sum(guessed) / len(guessed)) * 100}%")

# 5 примеров предсказывания
# rp = random.permutation(m)
# for i in range(5):
#     X2 = X[rp[i], :]
#     X2 = matrix(X[rp[i]])
#
#     pred = predict(X2.getA(), Theta1, Theta2)
#     pred = squeeze(pred)
#     displayData(X2, f"Neural Network Prediction: {pred} (digit {Y[rp[i]]})")

# Показ первых 100 ошибок
# errors = where(predicted != transpose(Y)[0])
# displayData(array([X[i] for i in errors[0][0:100]]), "Network errors")
