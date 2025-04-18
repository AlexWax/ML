import sklearn as sk
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_wine

X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', KNeighborsClassifier(n_neighbors=5))
])

pipe.fit(X_train, y_train)
score = pipe.score(X_test, y_test)

print(score)


def Scale(x):
    min = x.mean(axis=0)
    std = x.std(axis=0)

    X = (x - min)/std
    return X, min, std

a = np.array([1, 2, 3, 4, 5])

print(Scale(a))