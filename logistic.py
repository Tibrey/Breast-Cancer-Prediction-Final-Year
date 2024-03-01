import numpy as np
import pickle

class LogisticRegression:
    def __init__(self, learning_rate=0.0001):
        np.random.seed(1)
        self.learning_rate = learning_rate

    def initialize_parameter(self):
        self.W = np.zeros(self.X.shape[1])
        self.b = 0.0

    def forward(self, X):
        Z = np.matmul(X, self.W) + self.b
        A = sigmoid(Z)
        return A

    def compute_cost(self, predictions):
        m = self.X.shape[0]  # number of training examples
        # compute the cost
        cost = np.sum(
            (-np.log(predictions + 1e-8) * self.y)
            + (-np.log(1 - predictions + 1e-8)) * (1 - self.y)
        )  # adding small value epsilon to avoid log of 0
        cost = cost / m
        return cost

    def compute_gradient(self, predictions):
        # get training shape
        m = self.X.shape[0]

        # compute gradients
        self.dW = np.matmul(self.X.T, (predictions - self.y))
        self.dW = np.array([np.mean(grad) for grad in self.dW])

        self.db = np.sum(np.subtract(predictions, self.y))

        # scale gradients
        self.dW = self.dW * 1 / m
        self.db = self.db * 1

    def fit(self, X, y, iterations, plot_cost=True):
        self.X = X
        self.y = y

        self.initialize_parameter()

        costs = []
        for i in range(iterations):
            # forward propagation
            predictions = self.forward(self.X)

            # compute cost
            cost = self.compute_cost(predictions)
            costs.append(cost)

            # compute gradients
            self.compute_gradient(predictions)

            # update parameters
            self.W = self.W - self.learning_rate * self.dW
            self.b = self.b - self.learning_rate * self.db

            if i % 10000 == 0:
                print("Cost after iteration {}: {}".format(i, cost))

    def predict(self, X):
        predictions = self.forward(X)
        print(predictions)
        return np.round(predictions)

    def predict_proba_lr(self, X):
        decision_values = np.dot(X, self.W) + self.b
        probabilities = sigmoid(decision_values)
        return probabilities

    def save_model(self, filename=None):
        model_data = {"learning_rate": self.learning_rate, "W": self.W, "b": self.b}

        with open(filename, "wb") as file:
            pickle.dump(model_data, file)

    @classmethod
    def load_model(cls, filename):
        with open(filename, "rb") as file:
            model_data = pickle.load(file)

        # Create a new instance of the class and initialize it with the loaded parameters
        loaded_model = cls(model_data["learning_rate"])
        loaded_model.W = model_data["W"]
        loaded_model.b = model_data["b"]

        return loaded_model


def sigmoid(z):
    # Compute the sigmoid function using the formula: 1 / (1 + e^(-z)).
    sigmoid_result = 1 / (1 + np.exp(-z))

    # Return the computed sigmoid value.
    return sigmoid_result
