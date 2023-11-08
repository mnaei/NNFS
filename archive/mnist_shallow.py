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
    
class Loss:

    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss
    
class Loss_CategoricalCrossentropy(Loss):
    
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        correct_confidences = y_pred_clipped[range(samples), y_true]
        
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

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
    loss_activation = Activation_softmax_Loss_CategoricalCrossentropy()
    optimizer = Optimizer_SGD()


    for epoch in range(1000):

        dense1.forward(X)
        loss = loss_activation.forward(dense1.output, y)

        predictions = np.argmax(loss_activation.output, axis=1)
        accuracy = np.mean(predictions==y)

        if not epoch % 1:
            print('epoch:', epoch)
            print('\tLoss:', loss)
            print('\tAccuracy:', accuracy)

        loss_activation.backward(loss_activation.output, y)
        dense1.backward(loss_activation.dinputs)

        optimizer.update_params(dense1)

main()