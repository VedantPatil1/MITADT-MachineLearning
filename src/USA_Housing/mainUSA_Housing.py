from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


USAHousing = pd.read_csv('src/USA_Housing/USA_Housing.csv')
USAHousing.head()

# print(USAHousing.head())
# print(USAHousing.describe())
# print(USAHousing.info())

sns.pairplot(USAHousing)
sns.displot(USAHousing['Price'])
sns.heatmap(USAHousing.corr())
plt.show()
# print(USAHousing.columns)

X = USAHousing[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
                'Avg. Area Number of Bedrooms', 'Area Population', ]]
Y = USAHousing['Price']

x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.4, random_state=101)

print(x_train)
print(y_train)
