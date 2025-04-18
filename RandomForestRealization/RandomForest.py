import numpy as np
from sklearn.tree import DecisionTreeClassifier
np.random.seed(42)


class sample(object):
    def __init__(self, X, n_subspace):
        self.idx_subspace = self.random_subspace(X, n_subspace)

    def __call__(self, X, y):
        idx_obj = self.bootstrap_sample(X)
        X_sampled, y_sampled = self.get_subsample(X, y, self.idx_subspace, idx_obj)
        return X_sampled, y_sampled

    @staticmethod
    def bootstrap_sample(X):
        return np.unique(np.random.choice(len(X), [len(X),1], replace=True))

    @staticmethod
    def random_subspace(X, n_subspace):
        return np.random.choice(X.shape[1], size=n_subspace, replace=False)

    @staticmethod
    def get_subsample(X, y, idx_subspace, idx_obj):
        X_sampled = X[idx_obj, :]
        X_sampled = X_sampled[:, idx_subspace]
        y_sampled = y[idx_obj]
        return X_sampled, y_sampled


class RandomForest(object):
    def __init__(self, n_estimators: int, max_depth: int, subspaces_dim: int, random_state: int):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.subspaces_dim = subspaces_dim
        self.random_state = random_state
        self.subspace_idx = list()
        self.forest = list()

    def fit(self, X, y):
        for i in range(self.n_estimators):
            s = sample(X, self.subspaces_dim)

            bootstrap_indices = s.bootstrap_sample(X)
            self.subspace_idx.extend([s.idx_subspace])
            X_sampled, y_sampled = s.get_subsample(X, y, s.idx_subspace, bootstrap_indices)
            tree = DecisionTreeClassifier(max_depth=self.max_depth, random_state=self.random_state)

            tree.fit(X_sampled, y_sampled)
            self.forest.append(tree)

    def predict(self, X):
        res = np.ndarray(shape=[self.n_estimators, X.shape[0]])
        for i, elem in enumerate(self.forest):
            res[i, :] = elem.predict(X[:, self.subspace_idx[i]])
        return [round(elem) for elem in res.mean(axis=0)]
