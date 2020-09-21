# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 17:30:34 2020

@author: Asus
"""

# K-Means Clustering

#Projects: Customer Segmentation

#A Company wants to identify segments of customers for targetted marketing.



# Importing the libraries

#import numpy as np

import matplotlib.pyplot as plt

import pandas as pd



# Importing the dataset

dataset = pd.read_csv('UK-Bank.csv')

dataset.info()


X = dataset.iloc[:, [4,8]].values# independ vars should be conti vars preferbaly

# beacuse it is one d array we use array.rehspae(-1,1)

# y = dataset.iloc[:, 3].values

#X=x.reshape(-1,1)

# Splitting the dataset into the Training set and Test set

"""from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""



# Feature Scaling

"""from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)

X_test = sc_X.transform(X_test)

sc_y = StandardScaler()

y_train = sc_y.fit_transform(y_train)"""



# Using the elbow method to find the optimal number of clusters

from sklearn.cluster import KMeans

wcss = [] # EMPTY LIST DISTANCE BWT EACH DATA POINT FROM THEIR CENTROID
# Weighted cluster sq sum

for i in range(1, 20):

    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)

    kmeans.fit(X)

    wcss.append(kmeans.inertia_)

plt.plot(range(1, 20), wcss)

plt.title('The Elbow Method')

plt.xlabel('Number of clusters')

plt.ylabel('WCSS')

plt.show()

# from here you can calculate the optimum number of clusters required

# Fitting K-Means to the dataset

kmeans = KMeans(n_clusters = 4, init = 'k-means++', random_state = 42)

y_kmeans = kmeans.fit_predict(X)



kmeans = pd.DataFrame(y_kmeans)

dataset_1 = pd.concat([dataset,kmeans],axis=1)



# Visualising the clusters

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0,1], s = 10, c = 'red', label = 'Cluster 1')

plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1,1], s = 10, c = 'blue', label = 'Cluster 2')

plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2,1], s = 10, c = 'green', label = 'Cluster 3')

plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3,1], s = 10, c = 'black', label = 'Cluster 4')

#plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 10, c = 'magenta', label = 'Cluster 5')

#plt.scatter(X[y_kmeans == 5, 0], X[y_kmeans == 5, 1], s = 10, c = 'black', label = 'Cluster 6')


#plt.scatter(X[y_kmeans == 6, 0], X[y_kmeans == 6, 1], s = 10, c = 'orange', label = 'Cluster 7')


#plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')

plt.title('Clusters of customers')

plt.xlabel('Age')

plt.ylabel('Annual income')

plt.legend()

plt.show()