import numpy as np

def relu(x):
    return max(0, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x, axis=0)

def categorical_cross_entropy(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred))

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights1 = np.random.randint(-5, 5, (hidden_size, input_size))
        self.bias1 = np.random.randint(-5, 5, hidden_size)
        self.weights2 = np.random.randint(-5, 5, (output_size, hidden_size))
        self.bias2 = np.random.randint(-5, 5, output_size)

    def forward_pass(self, x):
        self.z1 = np.dot(self.weights1, x) + self.bias1
        self.a1 = np.vectorize(relu)(self.z1)
        self.z2 = np.dot(self.weights2, self.a1) + self.bias2
        self.a2 = softmax(self.z2)
        return self.a2

    def backpropagation(self, x, y_true, learning_rate):
        m = x.shape[1]
        dz2 = self.a2 - y_true
        dw2 = np.dot(dz2, self.a1.T) / m
        db2 = np.sum(dz2, axis=1, keepdims=True) / m
        dz1 = np.dot(self.weights2.T, dz2) * np.vectorize(lambda x: 1 if x > 0 else 0)(self.z1)
        dw1 = np.dot(dz1, x.T) / m
        db1 = np.sum(dz1, axis=1, keepdims=True) / m

        self.weights1 -= learning_rate * dw1
        self.bias1 -= learning_rate * db1.squeeze()
        self.weights2 -= learning_rate * dw2
        self.bias2 -= learning_rate * db2.squeeze()

    def train(self, x, y_true, epochs, learning_rate):
        for epoch in range(epochs):
            y_pred = self.forward_pass(x)
            loss = categorical_cross_entropy(y_true, y_pred)
            self.backpropagation(x, y_true, learning_rate)
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss}')

input_size = 2
hidden_size = 3
output_size = 2
nn = NeuralNetwork(input_size, hidden_size, output_size)

x = np.array([[1, 0], [0, 1]])
y_true = np.array([[1, 0], [0, 1]])

nn.train(x, y_true, epochs=1000, learning_rate=0.01)
