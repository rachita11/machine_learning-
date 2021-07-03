#This code uses K-Means clustering to graphically show the different classes of iris dataset.

#import necessary modules
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import metrics
from matplotlib import pyplot as plt

#bring dataset to your code
data = pd.read_csv("iris.csv")
data = data[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]

#perform dimensionality reduction to fit into 2D space
pca = PCA(2)
data1 = pca.fit_transform(data)

#build KMeans model
model = KMeans(n_clusters=3)
label = model.fit_predict(data1)

#checking the positions of all the centriods in each iteration
print(model.cluster_centers_)

#plotting the clusters formed
labels = np.unique(label)
for i in labels:
    plt.scatter(data1[label==i,0], data1[label==i,1])

#plotting the final centroids
centroids = model.cluster_centers_
plt.scatter(centroids[:,0] , centroids[:,1] ,marker='s', s = 100, color = 'k')

#labeling the clusters with the help of our iris dataset
plt.legend(['Iris-setosa','Iris-versicolor','Iris-virginica'])
plt.show()


#find the output and final graph in this file called "Iris part2"
