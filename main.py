import numpy as np
import math
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

def leaky_relu(x, alpha=0.02):
    return np.where(x >= 0, x, alpha * x)

x = np.linspace(-5, 5, 100)

sigmoid_y = sigmoid(x)
relu_y = relu(x)
tanh_y = tanh(x)
leaky_relu_y = leaky_relu(x)

plt.figure(figsize=(10, 6))
plt.plot(x, sigmoid_y, label='Sigmoid')
plt.plot(x, relu_y, label='ReLU')
plt.plot(x, tanh_y, label='Tanh')
plt.plot(x, leaky_relu_y, label='Leaky ReLU')
plt.title('Activation Functions')
plt.xlabel('Input')
plt.ylabel('Output')
plt.legend()
plt.grid(True)
plt.show()

random_values = [-3.5, -1.2, 0, 2.8, -4.1, 1.5, -0.7, 3.2, -2.4, 4.6]
for x in random_values: print(relu(x))
for x in random_values: print(tanh(x))
for x in random_values: print(leaky_relu(x))