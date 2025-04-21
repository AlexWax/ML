import numpy as np


class KNN_Classifier:
    """KNN-model realization"""
    def __init__(self, n_neighbors: int, **kwargs):
        """
        K - Global parameter characterise number of nearest neighbors
        """
        self.K = n_neighbors
        self.dots = []

    def fit(self, x: np.array, y: np.array) -> None:
        """
        Memorizing training selection
        x - array of objects: [obj, feature]
        y - array of true values for objects
        """
        self.dots = [(x[i], y[i]) for i in range(x.shape[0])]

    def predict(self, x: np.array) -> np.array:
        """
        Making prediction
        x - array of objects: [obj, feature]
        """
        predictions = []

        for elem in x:
            distances = [(obj[1], np.linalg.norm(obj[0] - elem))
                         for obj in self.dots]
            distances.sort(key=lambda elem: elem[1])

            count = {}
            for i in range(self.K):
                if distances[i][0] in count:
                    count[distances[i][0]] += 1
                else:
                    count[distances[i][0]] = 1
            max_value = max(count.items(), key=lambda m: m[1])[1]
            predictions.append(min([key for key, value in count.items() if value == max_value]))
        predictions = np.array(predictions)
        return predictions
