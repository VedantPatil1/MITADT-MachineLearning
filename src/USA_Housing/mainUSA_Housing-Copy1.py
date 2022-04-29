import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns

USAHousing = pd.read_csv('src/USA_Housing/USA_Housing.csv')
USAHousing.head()

# print(USAHousing.head())
# print(USAHousing.describe())
# print(USAHousing.info())


sns.displot(USAHousing['Price'])
sns.heatmap(USAHousing.corr())

print(USAHousing.columns)
