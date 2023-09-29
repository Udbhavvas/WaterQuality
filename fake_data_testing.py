# Imports
import SVM
import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib as plt

from sklearn.model_selection import train_test_split

##################
n_pts = 1000
np.random.seed(0)
Xa = np.array([np.random.normal(13, 2, n_pts),
              np.random.normal(12, 2, n_pts)]).T
Xb = np.array([np.random.normal(8, 2, n_pts),
              np.random.normal(6, 2, n_pts)]).T
 
X = np.vstack((Xa, Xb))
Y = np.matrix(np.append(np.zeros(n_pts), np.ones(n_pts))).T
 

plt.scatter(X[:n_pts,0], X[:n_pts,1])
plt.scatter(X[n_pts:,0], X[n_pts:,1])
########################




# # Instatiate SVM object
# SVM_instance = SVM.SVM(learning_rate=0.05, epochs=2000)

# #########################################################

# #fake data set
# X, y = datasets.make_blobs(
#     n_samples=2000, n_features=2, centers=2, cluster_std=5.0, random_state=100
# )
# y = np.where(y == 0, -1, 1)

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.5, random_state=123
# )



# SVM_instance.train(X_train, y_train)

# predictions = SVM_instance.predict(X_test)



# #matplotlib.

# print("SVM classification accuracy", SVM_instance.accuracy(y_test, predictions))

