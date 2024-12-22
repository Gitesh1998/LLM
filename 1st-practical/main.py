import numpy as np

class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.bias_hidden = np.zeros((1, hidden_size))
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        self.bias_output = np.zeros((1, output_size))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def forward(self, X):
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self.sigmoid(self.hidden_input)
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.final_output = self.sigmoid(self.final_input)
        return self.final_output
    
    def compute_loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)
    
# XOR dataset
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([[0],
              [1],
              [1],
              [0]])

input_size = 2
hidden_size = 2
output_size = 1

nn = SimpleNeuralNetwork(input_size, hidden_size, output_size)
output = nn.forward(X)
print("Initial Predictions:\n", output)

loss = nn.compute_loss(y, output)
print("Initial Loss:", loss)

import matplotlib.pyplot as plt

plt.scatter(X[:,0], X[:,1], c=y.flatten(), cmap='bwr')
plt.title("XOR Problem")
plt.xlabel("Input 1")
plt.ylabel("Input 2")
plt.show()