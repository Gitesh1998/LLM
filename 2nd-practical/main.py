import numpy as np
import matplotlib.pyplot as plt

# Define the function
def f(x):
    return (x - 3)**2

# Define the derivative of the function
def df(x):
    return 2 * (x - 3)

def gradient_descent_batch(initial_x, learning_rate, iterations):
    x = initial_x
    history = []
    for i in range(iterations):
        grad = df(x)
        x = x - learning_rate * grad
        history.append(x)
        print(f"Iteration {i+1}: x = {x}, f(x) = {f(x)}")
    return x, history

# Parameters
initial_x = 0.0
learning_rate = 0.1
iterations = 20

# Run Batch Gradient Descent
final_x, history = gradient_descent_batch(initial_x, learning_rate, iterations)