import numpy as np
import pandas as pd


class Layer_Dense:

    def __init__(self, inputs, neurons):
        # self.weights = 0.01 * np.random.randn(inputs, neurons)
        self.weights = 10 * np.random.randn(inputs, neurons)
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

class Loss:

    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_MeanSquaredError(Loss):
    
    def forward(self, y_pred, y_true):
        sample_losses = np.mean((y_true - np.squeeze(y_pred))**2)
        return sample_losses

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

    np.random.seed(1)

    df = pd.read_csv('./data/regression_train.csv')

    features = ['OverallQual' , 'GrLivArea' , 'TotalBsmtSF' , 'BsmtFinSF1' ,
            '2ndFlrSF'    , 'GarageArea', '1stFlrSF'    , 'YearBuilt'  ]

    X = df[features]
    y = df["SalePrice"]

    X = X.fillna(X.mean())

    # X = np.divide(X, 10)
    # y = np.divide(y, 10)

    X = X.values
    y = y.values

    dense1 = Layer_Dense(8, 8)
    activation1 = Activation_ReLU()
    dense2 = Layer_Dense(8, 1)
    activation2 = Activation_Linear()
    loss_mse = Loss_MeanSquaredError()
    optimizer = Optimizer_SGD()


    for epoch in range(2001):

        dense1.forward(X)
        activation1.forward(dense1.output)
        dense2.forward(activation1.output)
        activation2.forward(dense2.output)

        loss = loss_mse.calculate(activation2.output, y)

        if not epoch % 1000:
            print('epoch:', epoch)
            print('\tLoss:', loss)
            breakpoint()

        loss_mse.backward(activation2.output, y)
        activation2.backward(loss_mse.dinputs)
        dense2.backward(activation2.dinputs)
        activation1.backward(dense2.dinputs)
        dense1.backward(activation1.dinputs)

        optimizer.update_params(dense1)
        optimizer.update_params(dense2)

main()