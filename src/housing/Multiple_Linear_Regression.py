import pandas as pd
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv('src/housing/data_mv.csv')

print(dataset.shape)
print(dataset.head(5))

dataset.columns[dataset.isna().any()]

dataset.hours = dataset.hours.fillna(dataset.hours.mean())

X = dataset.iloc[:, :-1].values
print(X.shape)
print(X)

Y = dataset.iloc[:, -1].values
print(Y)

model = LinearRegression()
model.fit(X, Y)

a = [[9.2, 20, 0]]
PredictedmodelResult = model.predict(a)
print(PredictedmodelResult)
