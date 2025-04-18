from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression, Ridge, Lasso

diab = load_diabetes()

X = diab.data
y = diab.target

lr1 = LinearRegression()
lr1.fit(X, y)

for f, w in zip(diab.feature_names, lr1.coef_):
    print(f"{f:7s}: {w:6.2f}")

lr2 = Ridge(alpha=10.0)
lr2.fit(X, y)

for f, w in zip(diab.feature_names, lr2.coef_):
    print(f"{f:7s}: {w:6.2f}")

lr3 = Lasso(alpha=2.0)
lr3.fit(X, y)

for f, w in zip(diab.feature_names, lr3.coef_):
    print(f"{f:7s}: {w:6.2f}")