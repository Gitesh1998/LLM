import numpy as np


class SimpleNeuralNetworkGD:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        # Initialize weights and biases
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.bias_hidden = np.zeros((1, hidden_size))
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        self.bias_output = np.zeros((1, output_size))
        self.learning_rate = learning_rate
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def sigmoid_derivative(self, z):
        return z * (1 - z)
    
    def forward(self, X):
        # Forward propagation
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self.sigmoid(self.hidden_input)
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.final_output = self.sigmoid(self.final_input)
        return self.final_output
    
    def compute_loss(self, y_true, y_pred):
        # Mean Squared Error Loss
        return np.mean((y_true - y_pred) ** 2)
    
    def backward(self, X, y_true, y_pred):
        # Calculate the error
        error = y_pred - y_true
        d_output = error * self.sigmoid_derivative(y_pred)
        
        # Calculate error for hidden layer
        error_hidden = d_output.dot(self.weights_hidden_output.T)
        d_hidden = error_hidden * self.sigmoid_derivative(self.hidden_output)
        
        # Compute gradients
        grad_weights_hidden_output = self.hidden_output.T.dot(d_output)
        grad_bias_output = np.sum(d_output, axis=0, keepdims=True)
        grad_weights_input_hidden = X.T.dot(d_hidden)
        grad_bias_hidden = np.sum(d_hidden, axis=0, keepdims=True)
        
        # Update weights and biases
        self.weights_hidden_output -= self.learning_rate * grad_weights_hidden_output
        self.bias_output -= self.learning_rate * grad_bias_output
        self.weights_input_hidden -= self.learning_rate * grad_weights_input_hidden
        self.bias_hidden -= self.learning_rate * grad_bias_hidden
    
    def train(self, X, y, iterations):
        for i in range(iterations):
            y_pred = self.forward(X)
            loss = self.compute_loss(y, y_pred)
            self.backward(X, y, y_pred)
            if (i+1) % 1000 == 0 or i == 0:
                print(f"Iteration {i+1}: Loss = {loss}")
                
                
# XOR dataset
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([[0],
              [1],
              [1],
              [0]])

# Initialize the neural network
input_size = 2
hidden_size = 2
output_size = 1
learning_rate = 0.1
iterations = 10000

nn = SimpleNeuralNetworkGD(input_size, hidden_size, output_size, learning_rate)

# Train the neural network
nn.train(X, y, iterations)

# Perform predictions after training
output = nn.forward(X)
print("Final Predictions:\n", output)