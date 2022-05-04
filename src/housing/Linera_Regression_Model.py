import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


dataset = pd.read_csv('src/housing/dataset.csv')
print(dataset.shape)
print(dataset.head(5))

plt.xlabel('Area')
plt.ylabel('Price')
plt.scatter(dataset.area, dataset.price, color='red', marker='*')
plt.show()

X = dataset.drop('price', axis='columns')
print(X)

Y = dataset.price
print(Y)

model = LinearRegression()
model.fit(X, Y)

x = 40000
LandAreainSqFt = [[x]]
PredictedmodelResult = model.predict(LandAreainSqFt)
print(PredictedmodelResult)

m = model.coef_
print(m)

b = model.intercept_
print(b)

y = m*x + b
print("The Price of {0} Square feet Land is: {1}".format(x, y[0]))
