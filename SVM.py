import numpy as np
import pickle

class SVM:

    def __init__(self, iterations=1000, lr=0.01, lambdaa=0.01):
        self.lambdaa = lambdaa
        self.iterations = iterations
        self.lr = lr
        self.w = None
        self.b = None

    def initialize_parameters(self, X):
        m, n = X.shape
        self.w = np.zeros(n)
        self.b = 0


    def gradient_descent(self, X, y):
        y_ = np.where(y <= 0, -1, 1)
        for i, x in enumerate(X):
            if y_[i] * (np.dot(x, self.w) - self.b) >= 1:
                dw = 2 * self.lambdaa * self.w
                db = 0
            else:
                dw = 2 * self.lambdaa * self.w - np.dot(x, y_[i])
                db = y_[i]
            self.update_parameters(dw, db)

    def update_parameters(self, dw, db):
        self.w = self.w - self.lr * dw
        self.b = self.b - self.lr * db

    def fit(self, X, y):
        self.initialize_parameters(X)
        for i in range(self.iterations):
            self.gradient_descent(X, y)

    def predict(self, X):
        # get the outputs
        output = np.dot(X, self.w) - self.b
        print(output)
        print(type(output))
        print(type(self.w))
        print(type(self.b))
        print(self.w.tolist())
        print(self.b.tolist())
        # get the signs of the labels depending on if it's greater/less than zero
        label_signs = np.sign(output)
        # set predictions to 0 if they are less than or equal to -1 else set them to 1
        predictions = np.where(label_signs <= -1, 0, 1)
        return predictions

    def predict_proba_svm(self, X):
        decision_values = np.dot(X, self.w) - self.b
        probabilities = 1 / (1 + np.exp(-decision_values))
        return probabilities

    def save_model(self, filename=None):
        model_data = {
            "lambdaa": self.lambdaa,
            "learning_rate": self.lr,
            "W": self.w,
            "b": self.b,
        }
        print(type(self.w))

        with open(filename, "wb") as file:
            pickle.dump(model_data, file)

    @classmethod
    def load_model(cls, filename):

        with open(filename, "rb") as file:
            model_data = pickle.load(file)

        # Create a new instance of the class and initialize it with the loaded parameters
        loaded_model = cls(
            lr=model_data["learning_rate"], lambdaa=model_data["lambdaa"]
        )
        loaded_model.w = model_data["W"]
        loaded_model.b = model_data["b"]

        return loaded_model