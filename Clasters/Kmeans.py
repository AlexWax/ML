import numpy as np
import pandas as pd
import seaborn as sns
from collections import defaultdict


class KMeans(object):
    """
    KMeans algorithm realisation
    """
    def __init__(self, k: int, init: np.array):
        """
        :param k: number of clusters
        :param init: numpy array of initial cluster coordinates like: [clusters, features]
        """
        self.K = k
        self.cur_cor = init

    @staticmethod
    def points_clus(points: np.array, center_cor: np.array) -> dict:
        """
        We determine cluster for each plot, using the position of cluster`s centers
        :param points: list of objects like: [object, features]
        :param center_cor: numpy array of cluster coordinates like: [clusters, features]
        :return: dict of clusters and corresponding points like: {cluster: [point_features, ...], ...}
        """
        clusters = defaultdict(list)
        classes = []

        for elem in points:
            dist_to_class = []
            [dist_to_class.append(np.linalg.norm(elem - cl)) for cl in center_cor]
            classes.append(np.argmin(dist_to_class))

        for i, elem in enumerate(classes):
            clusters[elem].append(points[i])

        return clusters

    @staticmethod
    def find_centers(points):
        """
        We calculate and relocate center of each cluster
        :param points: dict of clusters and corresponding points like: {cluster: [point_features, ...], ...}
        :return: numpy array of updated cluster coordinates like: [clusters, features]
        """
        center_cor = []
        for key, value in points.items():
            cor = np.mean(value, axis=0)
            center_cor.append(cor)

        return np.array(center_cor)

    def fit(self, x: np.array, threshold: float = 0.001):
        """
        KMeans training. We determine optimal position of cluster`s centers
        :param x: list of objects like: [object, features]
        :param threshold: threshold to stop algorithm
        """
        offset = 1

        while offset > threshold:
            last_cor = self.cur_cor
            points = self.points_clus(x, last_cor)
            self.cur_cor = self.find_centers(points)
            offset = max([np.linalg.norm(elem1 - elem2) for elem1, elem2 in zip(self.cur_cor, last_cor)])

    def predict(self, x: np.array) -> np.array:
        """
        KMean prediction
        :param x: list of objects like: [object, features]
        :return: list (of length equal to x first dimension) with predicted classes to each object correspondingly
        """
        classes = []

        for elem in x:
            dist_to_class = []
            for cl in self.cur_cor:
                dist_to_class.append(np.linalg.norm(elem - cl))
            classes.append(np.argmin(dist_to_class))

        return np.array(classes)


if __name__ == "__main__":
    data1 = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Копия 1.csv')

    M = 2
    N = 2
    X = np.random.randn(N, M)

    model = KMeans(M, X)
    model.fit(data1[['a0', 'a1']].to_numpy())
    classses = model.predict(data1[['a0', 'a1']].to_numpy())

    sns.relplot(data=data1, x="a0", y="a1", hue=classses, palette='pastel')
