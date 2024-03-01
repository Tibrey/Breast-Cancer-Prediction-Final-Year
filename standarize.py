import pickle
import numpy as np

with open("D:/BreastCancerPrediction/Xtrain.pkl", "rb") as file:
    X_train = pickle.load(file)


def standardize_data(final_features):
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    final_features = (final_features - mean) / std
    return final_features
