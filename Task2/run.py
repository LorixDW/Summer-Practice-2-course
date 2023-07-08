from scipy import *
from numpy import *

import functions
from sigmoid import *
from cost_find import *
from predict import *
from functions import *

# Задание 1
X = io.loadmat("training_set.mat")['X']
Y = io.loadmat("training_set.mat")['y']

Theta1 = io.loadmat("weights.mat")['Theta1']
Theta2 = io.loadmat("weights.mat")['Theta2']

# Задание 2
input_layer_size = X.shape[1]
hidden_layer_size = Theta1.shape[0]
num_labels = Theta2.shape[0]
m = X.shape[0]
print(f"Структура нейросети: "
      f"\n\t-размер входного слоя - {input_layer_size};"
      f"\n\t-размер скрытого слоя - {hidden_layer_size};"
      f"\n\t-размер выходного слоя - {num_labels};")

# Задание 3
a1 = insert(X, 0, 1, axis=1)
a2 = insert(sigmoid(dot(a1, transpose(Theta1))), 0, 1, axis=1)
h = sigmoid(dot(a2, transpose(Theta2)))

y = zeros((m, 10))
for i in range(0, m):
    y[i][Y[i][0] - 1] = 1
print(f"Ошибка для alpha = 1: {cost(h, y, m, Theta1, Theta2, 1)}")
print(f"Ошибка для alpha = 0: {cost(h, y, m, Theta1, Theta2, 0)}")

# Задание 4
print(f"Произвоная сигмоиды в точках -1, 0.5, 0, 0.5, 1: {sigmoid_gradient(array([-1, 0.5, 0, 0.5, 1]))}")
theta1_rnd = random.rand(25, 401)
theta2_rnd = random.rand(10, 26)
pack = pack_teta(theta1_rnd, theta2_rnd)

print(f"Ошибка до обучения: {cost_alt(pack, X, y, m, 1)}")
pack_optimized = optimize.minimize(cost_alt, pack, method='L-BFGS-B', options={'maxiter': 1000},
                                   args=(X, y, m, 1), jac=functions.gradient)
print(f"Ошибка после обучения: {cost_alt(pack_optimized.x, X, y, m, 1)}")
theta1_opt, theta2_opt = unpack_teta(pack_optimized.x)
h_ = predict(X, theta1_opt, theta2_opt)
print(f"Эффективность обученной нейросети {sum(array([int(Y[i][0] == h_[i]) for i in range(0, m)])) / m * 100}%")
