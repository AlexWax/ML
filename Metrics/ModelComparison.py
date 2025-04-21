from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score


class ModelCompare:
    """
    Different models comparing with roc_auc and pr_auc metrics
    """
    def __init__(self, X_data, y_data):
        """
        X - matrix of objects: [obj, feature]
        y_true - true_value for x
        """
        self.X_train, self.x_test, self.y_train, self.y_test = train_test_split(X_data, y_data, test_size=0.3, random_state=42)

    def auc_(self, model):
        """
        Metrics calculation
        :param model: ML-model
        """
        pipe = Pipeline([
          ('scaler', StandardScaler()),
          ('classifier', model)
        ])
        pipe.fit(self.X_train, self.y_train)
        score = pipe.predict_proba(self.x_test)[:,1]

        precision, recall, _ = precision_recall_curve(self.y_test, score)
        auc_pr_model = auc(recall, precision)

        roc_auc_model = roc_auc_score(self.y_test, score)
        return auc_pr_model, roc_auc_model

    def predicts(self, models: list) -> dict:
        """
        Results
        :param models: list of ML-models
        """
        results = {str(model): self.auc_(model) for model in models}
        return results


if __name__ == '__main__':
    from sklearn.datasets import fetch_openml

    data = fetch_openml(data_id=42608)
    X, y = data['data'].drop(columns='Outcome').values, data['data']['Outcome'].astype(int).values

    tree = DecisionTreeClassifier(random_state=42)
    lr = LogisticRegression(random_state=42)
    knn = KNeighborsClassifier()
    svm = SVC(probability=True, random_state=42)
    models = [tree, lr, knn, svm]

    MC = ModelCompare(X, y)
    results = MC.predicts(models)

    max(results.items(), key=lambda x: x[1][0])
    max(results.items(), key=lambda x: x[1][1])