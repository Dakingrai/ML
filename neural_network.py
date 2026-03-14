import numpy as np
import pdb

class TwoLayerNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):

        self.W1 = np.random.randn(input_size, hidden_size) *  np.sqrt(1 / input_size)
        self.b1 = np.zeros((1, hidden_size))

        self.W2 = np.random.randn(hidden_size, output_size) *  np.sqrt(1 / hidden_size)
        self.b2 = np.zeros((1, output_size))

        self.lr = learning_rate

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)

        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.softmax(self.z2)

        return self.a2

    def sigmoid(self, z):
        z_clip = np.clip(z, -500, 500)
        return 1/(1+np.exp(-z_clip))

    def backward(self, X, y):
        # calculate gradient
        dz2 = self.a2 - y
        num_samples = X.shape[0]
        dw2 = np.dot(self.a1.T, dz2) / num_samples
        db2 = np.sum(dz2, axis=0, keepdims=True)/num_samples

        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self.sigmoid_derivative(self.a1) # intermediate error
        dw1 = np.dot(X.T, dz1)/num_samples
        db1 = np.sum(dz1, axis=0, keepdims=True)/num_samples

        self.W2 -= self.lr * dw2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dw1
        self.b1 -= self.lr * db1

    def compute_loss(self, y_true, y_pred):
        m = y_true.shape[0]
        # Cross-entropy loss with epsilon to avoid log(0)
        loss = -np.sum(y_true * np.log(y_pred)) / m
        return loss

    def train_step(self, X, y_true):
        y_pred = self.forward(X)
        loss = self.compute_loss(y_true, y_pred)
        self.backward(X, y_true)
        return loss

    def softmax(self, z):
        # Numerical stability: shift z by subtracting the max value
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def sigmoid_derivative(self, a):
        # 'a' is the output of sigmoid(z)
        return a * (1 - a)

    def predict(self, X):
        y_pred = self.forward(X)
        return np.argmax(y_pred, axis=1)

def calculate_accuracy(y_pred, y_true):
    
    # Convert one-hot y_true to indices: [[1, 0], [0, 1]] -> [0, 1]
    y_true_indices = np.argmax(y_true, axis=1)
    
    # Calculate percentage of correct matches
    accuracy = np.mean(y_pred == y_true_indices)
    return accuracy * 100  # Return as a percentage

if __name__ == "__main__":    
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    # One-hot encoded labels: 0 -> [1, 0], 1 -> [0, 1]
    y = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])

    # Initialize network
    input_size = 2
    hidden_size = 4
    output_size = 2

    nn = TwoLayerNeuralNetwork(input_size, hidden_size, output_size, learning_rate=0.5)

    for epoch in range(5001):
        loss = nn.train_step(X, y)
        if epoch % 1000 == 0:
            print(f"Epoch {epoch} | Loss: {loss:.4f}")

    print("\nPredictions:", nn.predict(X))
    y_pred = nn.predict(X)
    print("Accuracy:", calculate_accuracy(y_pred, y))