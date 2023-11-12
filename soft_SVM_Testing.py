# Imports
import SVM
import Soft_SVM
import numpy as np
import pandas as pd
from sklearn import datasets
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split


##################
n_pts = 50
np.random.seed()
Xa = np.array([np.random.normal(9, 1.25, n_pts),
              np.random.normal(9, 1.25, n_pts)]).T
Xb = np.array([np.random.normal(12, 1.25, n_pts),
              np.random.normal(12, 1.25, n_pts)]).T

X = np.vstack((Xa, Xb))
negative_ones = np.full((1, n_pts), -1)
#print(y)
positive_ones = np.full((1, n_pts), 1)
Y = np.append(negative_ones, positive_ones)
 

#Class 1 X and Y values

Class1 = [X[:n_pts,0], X[:n_pts,1]]
plt.scatter(Class1[0], Class1[1])
#plt.scatter(X[:n_pts,0], X[:n_pts,1])

#CLass 2 X and Y values

Class2 = [X[n_pts:,0], X[n_pts:,1]]
plt.scatter(Class2[0], Class2[1])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=123, shuffle = True )


print("SDSDSDDS")

# plt.scatter(X[n_pts:,0], X[n_pts:,1])
# plt.show()


#Class1[0] Class2[0]
soft_svm = Soft_SVM.SoftSVM(learning_rate=0.01, epochs=10000)
print("REACHED THIS FAR1")
soft_svm.train(X_train,Y_train)
print("REACHED THIS FAR2")
y_pred = soft_svm.predict(X_test)
print(f"accuracy = {soft_svm.accuracy(Y_test, y_pred)}")



print(soft_svm.w)


b = soft_svm.w[0]
w1 = soft_svm.w[1]
w2 = soft_svm.w[2]


print("REACHED THIS FAR3")
print(f"W1: {w1}")
print(f"W2: {w2}")
print(f"B: {b}")


# Plot svm line
y_start = (-1  * w1/w2) * (-5) - b/w2
y_end = (-1 * w1/w2) * (20) - b/w2


########################
plt.plot([-5, 20], [y_start, y_end])

#plt.Axes.plot([-5, 20], [y_start, y_end], linestyle='-', color='blue', label='My Line')

plt.savefig("plot.png")
