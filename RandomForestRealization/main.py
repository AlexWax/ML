import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from RandomForest import RandomForest

irises = load_iris()
X = irises['data']
y = irises['target']
X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

size = 11
acc = np.ndarray(shape=[size,size])
mp = np.linspace(1,51,size)

for i in range(size):
    for j in range(size):
        rf = RandomForest(int(i), int(j), 3, 42)
        rf.fit(X_train, y_train)
        predict = rf.predict(x_test)
        acc[i, j] = accuracy_score(predict, y_test)


fig, ax = plt.subplots()

ax.pcolormesh(mp, mp, acc, vmin=0, vmax=1.0)

plt.show()