# Imports
import pandas as pd
import numpy as np
import SVM_OLD
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# Obtaining/Cleaning CSV file
columns = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity', 'Potability']
data = pd.read_csv('water_potability.csv', names=columns, skiprows=1)

# Drop rows with NaN values
data = data.dropna()

pd.set_option('display.max_colwidth', None)

# Prints out cleaned data
data.head()

#randomizes data points
data = data.sample(frac=1)

#drops potability column from the data read in from csv
pot = data["Potability"]
data = data.drop("Potability", axis=1)

#Places data minus potability into X and potability data into Y
# X = np.array(data)
# y = np.array(pot)

# Reassign labels as -1 and 1
# y = np.where(y == 0, -1, 1)

ourSVM = SVM_OLD.SVM()

#fake data set
X, y = datasets.make_blobs(
    n_samples=2000, n_features=8, centers=2, cluster_std=5.0, random_state=100
)
y = np.where(y == 0, -1, 1)

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.5, random_state=123
# )



ourSVM.set_weights(X)

num_sections = 8

X_split_array = np.array_split(X, num_sections)
y_split_array = np.array_split(y, num_sections)

def visualize_svm():
        def get_hyperplane_value(x, w, b, offset):
            return (-w[0] * x + b + offset) / w[1]

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.scatter(X[:, 0], X[:, 1], marker="o", c=y)

        x0_1 = np.amin(X[:, 0])
        x0_2 = np.amax(X[:, 0])

        x1_1 = get_hyperplane_value(x0_1, ourSVM.w, ourSVM.b, 0)
        x1_2 = get_hyperplane_value(x0_2, ourSVM.w, ourSVM.b, 0)

        x1_1_m = get_hyperplane_value(x0_1, ourSVM.w, ourSVM.b, -1)
        x1_2_m = get_hyperplane_value(x0_2, ourSVM.w, ourSVM.b, -1)
        x1_1_p = get_hyperplane_value(x0_1, ourSVM.w, ourSVM.b, 1)
        x1_2_p = get_hyperplane_value(x0_2, ourSVM.w, ourSVM.b, 1)

        ax.plot([x0_1, x0_2], [x1_1, x1_2], "y--")
        ax.plot([x0_1, x0_2], [x1_1_m, x1_2_m], "k")
        ax.plot([x0_1, x0_2], [x1_1_p, x1_2_p], "k")

        x1_min = np.amin(X[:, 1])
        x1_max = np.amax(X[:, 1])
        ax.set_ylim([x1_min - 3, x1_max + 3])

        plt.show()
        return



j = 0
while j < num_sections:
      test_section = j
      print(ourSVM.w)
      X_train = X_split_array[0]
      if test_section == 0:
            X_train = X_split_array[1]

      for index, i in enumerate(X_split_array):
            if(index != test_section):
                  X_train = np.concatenate((X_train,i),axis = 0)
      X_test = X_split_array[test_section]

      y_train = y_split_array[0]
      if test_section == 0:
            y_train = y_split_array[1]

      for index, i in enumerate(y_split_array):
            if(index != test_section):
                   y_train = np.concatenate((y_train,i), axis = 0)   
      y_test = y_split_array[test_section]
     
      ourSVM.fit(X_train, y_train)

      predictions = ourSVM.predict(X_test)
      print(ourSVM.w)
      print("SVM classification accuracy", ourSVM.accuracy(y_test, predictions))
      j += 1
      
visualize_svm()


print("SVM classification accuracy", ourSVM.accuracy(y_test, predictions))

