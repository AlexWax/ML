import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
np.random.seed(42)


def pca_ideal_comp(X, y, components, model):
    X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    acc = dict()
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    x_test = scaler.transform(x_test)

    for elem in components:

        pca = PCA(n_components=elem)
        X_tr = pca.fit_transform(X_train)
        x_ts = pca.transform(x_test)

        lr.fit(X_tr, y_train)
        predict = lr.predict(x_ts)
        acc[elem] = accuracy_score(predict, y_test)

    return max(acc.items(), key= lambda x: x[1])


if __name__ == '__main__':

    from sklearn.datasets import fetch_openml
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LogisticRegression

    mnist = fetch_openml('mnist_784')
    X = mnist.data.to_numpy()
    y = mnist.target.to_numpy()
    part = 2000
    X = X[:part]
    y = y[:part]

    plt.figure(figsize=(20, 4))
    for index, (image, label) in enumerate(zip(X[0:5], y[0:5])):
        plt.subplot(1, 5, index + 1)
        plt.imshow(np.reshape(image, (28, 28)), cmap=plt.cm.gray)
        plt.title('Training: %s\n' % label, fontsize=20)

    N_COMPONENTS = [1,3,5,10,15,20,30,40,50,60]
    lr = LogisticRegression(max_iter=500)

    best_result = pca_ideal_comp(X, y, N_COMPONENTS, model=lr)
