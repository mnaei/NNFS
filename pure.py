import numpy as np
import nnfs
from nnfs.datasets import spiral_data, vertical_data

#np.zeros

def zeros(size):
    z = []
    for i in range(size):
        z.append(0)
    z = [z]
    return np.asarray(z, dtype=np.float32)

# #np.dot

# def matrix_matrix_multiplication(matrix_a, matrix_b):
#     assert len(matrix_a[0]) == len(matrix_b)
#     matrix_c = []
#     for i in range(len(matrix_a)):
#         matrix_c.append([])
#         for j in range(len(matrix_b[0])):
#             matrix_c[i].append(0)
#             for k in range(len(matrix_b)):
#                 matrix_c[i][j] += matrix_a[i][k] * matrix_b[k][j]
#     return matrix_c

# def matrix_vector_multiplication(matrix_a, vector_b):
#     assert len(matrix_a[0]) == len(vector_b)
#     vector_c = []
#     for i in range(len(matrix_a)):
#         vector_c.append(0)
#         for j in range(len(vector_b)):
#             vector_c[i] += matrix_a[i][j] * vector_b[j]
#     return vector_c

# def matrix_vector_addition(matrix_a, vector_b):
#     assert len(matrix_a[0]) == len(vector_b)
#     matrix_c = []
#     for i in range(len(matrix_a)):
#         matrix_c.append(zeros(len(vector_b)))
#         for j in range(len(vector_b)):
#             matrix_c[i][j] += matrix_a[i][j] + vector_b[j]
#     return matrix_c 

# #np.T

# def transpose(matrix):
#     matrix_t = []
#     for i in range(len(matrix[0])):
#         matrix_t.append([])
#         for j in range(len(matrix)):
#             matrix_t[i].append(matrix[j][i])
#     return matrix_t

#np.sum

def sum_cols(matrix):
    matrix_s = []
    for i in range(len(matrix[0])):
        matrix_s.append(0)
        for j in range(len(matrix)):
            matrix_s[i] += matrix[j][i]
    matrix_s = [matrix_s]
    return np.asarray(matrix_s)

def sum_rows(matrix):
    matrix_s = []
    for r in matrix:
        matrix_s.append(sum(r))
    matrix_s = [matrix_s]
    return np.asarray(matrix_s)

#np.maximum

maximum = lambda x: x if x > 0 else 0.0

def maximum_vector(vector):
    vector_m = []
    for i, r in enumerate(vector):
        vector_m.append([])
        for c in r:
            vector_m[i].append(maximum(c))
            # breakpoint()
    return np.asarray(vector_m, dtype=object)


#np.exp

#np.mean

#np.argmax

#np.clip

#np.eye

#np.diagflat

#np.empty_like

#np.log


class Layer_Dense:

    def __init__(self, inputs, neurons):
        self.weights = 0.01 * np.random.randn(inputs, neurons)
        self.biases = zeros(neurons)

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
    
    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = sum_cols(dvalues)
        self.dinputs = np.dot(dvalues, self.weights.T)

class Activation_ReLU:
    
    def forward(self, inputs):
        self.input = inputs
        self.output = maximum_vector(inputs)
    
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.input <= 0] = 0

class Activation_Softmax:

    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
    
    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)

        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

class Loss:

    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss
    
class Loss_CategoricalCrossentropy(Loss):
    
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)
        
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
    
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])

        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        
        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / samples

class Activation_softmax_Loss_CategoricalCrossentropy:
    
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()
    
    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true)
    
    def backward(self, dvalues, y_true):
        samples = len(dvalues)

        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples

class Optimizer_SGD:

    def __init__(self, learning_rate=1.0):
        self.learning_rate = learning_rate
    
    def update_params(self, layer):
        layer.weights += -self.learning_rate * layer.dweights
        layer.biases += -self.learning_rate * layer.dbiases
    

def main():

    nnfs.init()
    np.random.seed(0)
    
    X, y = spiral_data(samples=100, classes=3)

    dense1 = Layer_Dense(2, 64)
    activation1 = Activation_ReLU()
    dense2 = Layer_Dense(64, 3)
    loss_activation = Activation_softmax_Loss_CategoricalCrossentropy()

    optimizer = Optimizer_SGD()


    for epoch in range(10001):

        dense1.forward(X)
        activation1.forward(dense1.output)
        dense2.forward(activation1.output)
        loss = loss_activation.forward(dense2.output, y)


        predictions = np.argmax(loss_activation.output, axis=1)
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)        
        accuracy = np.mean(predictions==y)

        if not epoch % 10:
            print('epoch:', epoch)
            print('\tLoss:', loss)
            print('\tAccuracy:', accuracy)

        loss_activation.backward(loss_activation.output, y)
        dense2.backward(loss_activation.dinputs)
        activation1.backward(dense2.dinputs)
        dense1.backward(activation1.dinputs)

        optimizer.update_params(dense1)
        optimizer.update_params(dense2)

main()