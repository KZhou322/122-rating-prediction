import numpy as np 
import sklearn.cluster as cluster
import matplotlib.pyplot as plt
import pandas as pd


untrimmed_data = pd.read_csv("user_history.csv")
data = untrimmed_data.to_numpy()[:,1:]



k=3 # number of clusters
n=10 # number of singular values to keep
clusters = cluster.KMeans(n_clusters=k, n_init=100, max_iter=100)
cluster_labels = clusters.fit_predict(data)
centroids = clusters.cluster_centers_
ids = data[:,0]

np.save("cluster_labels", cluster_labels)
np.save("centroids", centroids)
np.save("ids", ids)


_, _, VT = np.linalg.svd(data)
data_projected = data @ (VT[:n,:].T)
centroids_projected = centroids @ (VT[:n,:].T)

plt.scatter(data_projected[:,0], data_projected[:,1], c=cluster_labels, cmap="viridis")
plt.scatter(centroids_projected[:,0], centroids_projected[:,1], c='red')
plt.show()






errors = []
for k in range(1,11):
    clusters = cluster.KMeans(n_clusters=k)
    cluster_labels = clusters.fit_predict(data)
    centroids = clusters.cluster_centers_
    error = 0.0
    for i in range(np.size(data, axis=0)):
        error += np.linalg.norm(data[i,:] - centroids[cluster_labels[i]]) ** 2
    error /= np.size(data, axis=0)
    errors.append(error)

plt.plot(list(range(1,11)), errors)
print(errors)
plt.title("Clustering error")
plt.xlabel("number of clusters")
plt.ylabel("MSE")
plt.show()
