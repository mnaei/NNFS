import numpy as np
import pandas as pd

class Layer_Dense:

    def __init__(self, inputs, neurons):
        self.weights = np.random.randint(-500, 501, size=(inputs, neurons))
        self.biases = np.zeros((1, neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
    
    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)

class Activation_Softmax:

    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
    
    def backward(self, dvalues, y_true):
        samples = len(dvalues)

        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples

class Optimizer_SGD:
    
    def update_params(self, layer):
        layer.weights += -layer.dweights.astype(np.int64) 
        layer.biases += -layer.dbiases.astype(np.int64) 
    
def main():

    np.random.seed(0)

    df = pd.read_csv('./data/mnist_train.csv')

    X = df.iloc[:, 1:].values
    y = df.iloc[:, 0].values

    dense1 = Layer_Dense(784, 10)
    activation1 = Activation_Softmax()
    optimizer = Optimizer_SGD()


    for epoch in range(81):

        dense1.forward(X)
        activation1.forward(dense1.output)

        predictions = np.argmax(activation1.output, axis=1)
        accuracy = np.mean(predictions==y)

        print('epoch:', epoch, 'Accuracy:', accuracy)

        activation1.backward(activation1.output, y)
        dense1.backward(activation1.dinputs)

        optimizer.update_params(dense1)

main()