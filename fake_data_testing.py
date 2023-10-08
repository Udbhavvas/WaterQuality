# Imports
import SVM
import numpy as np
import pandas as pd
from sklearn import datasets
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split


##################
n_pts = 1000
np.random.seed(0)
Xa = np.array([np.random.normal(15, 1.75, n_pts),
              np.random.normal(15, 1.75, n_pts)]).T
Xb = np.array([np.random.normal(8, 1.75, n_pts),
              np.random.normal(6, 1.75, n_pts)]).T

X = np.vstack((Xa, Xb))
negative_ones = np.full((1, n_pts), -1)
#print(y)
positive_ones = np.full((1, n_pts), 1)
Y = np.matrix(np.append(negative_ones, positive_ones)).T
 

#Class 1 X and Y values

Class1 = [X[:n_pts,0], X[:n_pts,1]]
plt.scatter(Class1[0], Class1[1])
#plt.scatter(X[:n_pts,0], X[:n_pts,1])

#CLass 2 X and Y values

Class2 = [X[n_pts:,0], X[n_pts:,1]]
plt.scatter(Class2[0], Class2[1])
# plt.scatter(X[n_pts:,0], X[n_pts:,1])
plt.show()


#Class1[0] Class2[0]
svm = SVM.SVM()
svm.train(X,Y)
y_pred = svm.predict(X)
print(svm.accuracy(Y, y_pred))



print(svm.w)
w1 = svm.w[0]
w2 = svm.w[1]

b = svm.b


y_start = (-w1/w2) * (-5) + b/w2
y_end = (-w1/w2) * (20) + b/w2


########################


plt.Axes.plot([-5, 20], [y_start, y_end], linestyle='-', color='blue', label='My Line')

plt.savefig("plot.png")

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

