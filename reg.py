import numpy as np
import pandas as pd


class Layer_Dense:

    def __init__(self, inputs, neurons):
        self.weights = 0.01 * np.random.randn(inputs, neurons)
        self.biases = np.zeros((1, neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
    
    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)

class Activation_ReLU:
    
    def forward(self, inputs):
        self.input = inputs
        self.output = np.maximum(0, inputs)
    
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.input <= 0] = 0

class Activation_Linear:
        
    def forward(self, inputs):
        self.input = inputs
        self.output = inputs
    
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()


class Loss_MeanSquaredError():

    def calculate(self, output, y):
        sample_losses = np.mean((y - np.squeeze(output))**2)
        data_loss = np.mean(sample_losses)
        return data_loss

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        outputs = len(dvalues[0])

        self.dinputs = -2 * (y_true - np.squeeze(dvalues)) / outputs
        self.dinputs = self.dinputs / samples
        self.dinputs = self.dinputs.reshape((samples, 1))

class Optimizer_SGD:

    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate
    
    def update_params(self, layer):
        layer.weights += -self.learning_rate * layer.dweights
        layer.biases += -self.learning_rate * layer.dbiases

class Accuracy_Regression:
        
        def calculate(self, predictions, y):
            comparisons = self.compare(predictions, y)
            accuracy = np.mean(comparisons)
            return accuracy
        
        def compare(self, predictions, y):
            return predictions == y 

def main():

    np.random.seed(0)

    df = pd.read_csv('./data/regression_train.csv')

    features = ['OverallQual' , 'GrLivArea' , 'TotalBsmtSF' , 'BsmtFinSF1' ,
            '2ndFlrSF'    , 'GarageArea', '1stFlrSF'    , 'YearBuilt'  ]

    X = df[features]
    y = df["SalePrice"]

    X = X.fillna(X.mean())

    X = X / X.max()
    y = y / y.max()

    X = X.values
    y = y.values

    dense1 = Layer_Dense(8, 32)
    activation1 = Activation_ReLU()
    dense2 = Layer_Dense(32, 1)
    activation2 = Activation_Linear()
    loss_mse = Loss_MeanSquaredError()
    optimizer = Optimizer_SGD()


    for epoch in range(11):

        dense1.forward(X)
        activation1.forward(dense1.output)
        dense2.forward(activation1.output)
        activation2.forward(dense2.output)

        loss = loss_mse.calculate(activation2.output, y)

        if not epoch % 1:
            print('epoch:', epoch)
            print('\tLoss:', loss)

        loss_mse.backward(activation2.output, y)
        activation2.backward(loss_mse.dinputs)
        dense2.backward(activation2.dinputs)
        activation1.backward(dense2.dinputs)
        dense1.backward(activation1.dinputs)

        optimizer.update_params(dense1)
        optimizer.update_params(dense2)

main()