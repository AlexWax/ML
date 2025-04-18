from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier

data = load_breast_cancer()
X = data.data
y = data.target
tree1 = DecisionTreeClassifier(criterion='gini')
tree1.fit(X, y)

print(tree1.min_weight_fraction_leaf)