import numpy as np

# Sigmoid function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Input dataset
inputs = np.array([[0,0,1],
                   [1,1,1],
                   [1,0,1],
                   [0,1,1]])

# Output dataset
outputs = np.array([[0,1,1,0]]).T

# Seed the random number generator
np.random.seed(1)

# Initialize weights with random values
synaptic_weights_1 = 2 * np.random.random((3,4)) - 1
synaptic_weights_2 = 2 * np.random.random((4,1)) - 1

# Main training loop
for iteration in range(10000):

    # Forward propagate through layers
    layer1 = sigmoid(np.dot(inputs, synaptic_weights_1))
    layer2 = sigmoid(np.dot(layer1, synaptic_weights_2))

    # Calculate the error
    layer2_error = outputs - layer2

    # Backpropagate the error
    layer2_delta = layer2_error * sigmoid_derivative(layer2)
    layer1_error = layer2_delta.dot(synaptic_weights_2.T)
    layer1_delta = layer1_error * sigmoid_derivative(layer1)

    # Update the weights
    synaptic_weights_2 += layer1.T.dot(layer2_delta)
    synaptic_weights_1 += inputs.T.dot(layer1_delta)

print("Output after training:")
print(layer2)