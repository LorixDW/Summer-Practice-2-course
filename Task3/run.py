from numpy import *
import matplotlib.pyplot as plt
from normalDistribution import *

size = 10000
data = genetrate(size, mu=0, sigma=1)

# Гистонрамма частот
# plt.hist(data, bins=300)
# plt.xlabel("Значение")
# plt.ylabel("Частота")
# plt.title("Частоты выпавших значений")
# plt.show()


print(f"Точечная оценка мат. ожидания выборки: {sampleAvg(data)}")
print(f"Точечная оценка дисперсии выборки: {sampleVarience(data)}")
print(f"Точечная оценка СКО: {sampleStandartDeviation(data)}")
params = paramsMLE(data)
print(f"Оценка параметров по Методу максимального правдоподобия"
      f"\nMu: {params[0]}\nSigma: {params[1]}")

x_min = -5.0
x_max = 5.0

x = arange(x_min, x_max, 0.01)

fig, ax = plt.subplots(1, 2, figsize=(8, 4))

ax[0].plot(x, [densityFunc(x[i], params[0], params[1]) for i in range(len(x))])
ax[0].set_title('Плотность вероятности')
ax[0].set_xlabel('Значение')
ax[0].set_ylabel('Плотность')
ax[0].axis([x_min, x_max, 0, 1])

ax[1].plot(x, [distributinFunc(x[i], params[0], params[1]) for i in range(len(x))])
ax[1].set_title('Функция распределения')
ax[1].set_xlabel('Значение')
ax[1].set_ylabel('Вероятность')
ax[1].axis([x_min, x_max, 0, 1])

plt.show()