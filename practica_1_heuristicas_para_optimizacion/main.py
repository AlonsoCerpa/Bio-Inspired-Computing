import random
import math
import matplotlib.pyplot as plt

def neighbor(x, left, right):
    delta = random.uniform(left/10.0, right/10.0)
    x_new = x + delta
    while x_new < left or x_new > right:
        delta = random.uniform(left/10.0, right/10.0)
        x_new = x + delta
    return x_new

def hill_climbing(n, func, left, right):
    y_array = []
    i = 0
    x1 = random.uniform(left, right)
    x2 = random.uniform(left, right)
    while i < n:
        x1_new = neighbor(x1, left, right)
        x2_new = neighbor(x2, left, right)
        y = func(x1, x2)
        y_array.append(y)
        y_new = func(x1_new, x2_new)
        if y_new > y:
            x1 = x1_new
            x2 = x2_new
        i += 1
    return x1, x2, y_array

def simulated_annealing(n, max_temperature, func, left, right):
    dec = max_temperature / n
    y_array = []
    t = max_temperature
    x1 = random.uniform(left, right)
    x2 = random.uniform(left, right)
    while t > 0:
        y = func(x1, x2)
        y_array.append(y)
        x1_new = neighbor(x1, left, right)
        x2_new = neighbor(x2, left, right)
        y_new = func(x1_new, x2_new)
        d_y = y_new - y
        if d_y > 0:
            x1 = x1_new
            x2 = x2_new
        elif math.exp(d_y / t) > random.uniform(0.0, 1.0):
            x1 = x1_new
            x2 = x2_new
        t -= dec
    return x1, x2, y_array

def f1(x1, x2):
    return pow(100 * (pow(x1, 2) - x2), 2) + pow(1 - x1, 2)

#parametros
num_iter = 100
left = -2.048
right = 2.048
max_temperature = 10000

"""
#Prueba 1
num_pruebas = 100
cont_a = 0
cont_b = 0
for i in range(num_pruebas):
    x1_a, x2_a, y_array_a = hill_climbing(num_iter, f1, left, right)
    y_a = f1(x1_a, x2_a)

    x1_b, x2_b, y_array_b = simulated_annealing(num_iter, max_temperature, f1, left, right)
    y_b = f1(x1_b, x2_b)

    if (y_a > y_b):
        cont_a += 1
    else:
        cont_b += 1

print("Hill Climbing fue mejor en ", cont_a, " de ", num_pruebas, " ocasiones.")
print("Simulated Annealing fue mejor en ", cont_b, " de ", num_pruebas, " ocasiones.")
"""


x1_a, x2_a, y_array_a = hill_climbing(num_iter, f1, left, right)
print("Hill climbing")
print("x1 = ", x1_a)
print("x2 = ", x2_a)
print("y = ", f1(x1_a, x2_a))
plt.plot(y_array_a)
plt.ylabel('energía')
plt.xlabel('iteración')
plt.show()

print("\n")

x1_b, x2_b, y_array_b = simulated_annealing(num_iter, max_temperature, f1, left, right)
print("Simulated annealing")
print("x1 = ", x1_b)
print("x2 = ", x2_b)
print("y = ", f1(x1_b, x2_b))
plt.plot(y_array_b)
plt.ylabel('energía')
plt.xlabel('iteración')
plt.show()