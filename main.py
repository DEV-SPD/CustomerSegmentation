import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# Loading the dataset
df = pd.read_csv('Mall_Customers.csv')
print(df.head())

# Analysis
print(df.shape)
print(df.isnull().sum())
print(df.describe())

# Choosing Annual Income & SpendingScore Parameters
X = df.iloc[:, 3:5].values
print(X)

# Applying Elbow Method to determine number of clusters
sse = []
for i in range(1, 11):
    model = KMeans(n_clusters=i)
    model.fit(X)
    sse.append(model.inertia_)

plt.plot(range(1, 11), sse)
plt.xlabel('Values of K')
plt.ylabel('SSE')
plt.title('PLOT TO FIND ELBOW POINT')
plt.show()
# k = 5 ie optimum no. of cluster as per elbow point method is 5

# Training the model
model = KMeans(n_clusters=5)
y_predicted = model.fit_predict(X)
df['clusters'] = y_predicted
print(df.head())





