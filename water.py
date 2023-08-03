# Imports
import pandas as pd
import numpy as np
import SVM


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
X = np.array(data)
y = np.array(pot)

# Reassign labels as -1 and 1
y = np.where(y == 0, -1, 1)


# Split data into 4 blocks, concatenate first 3 blocks for training and use D for testing
X = np.array(X)
y = np.array(y)

num_sections = 4
test_section = 2

split_array = np.array_split(X, num_sections)
for i in split_array:
      if(i != test_section):
            X_train = np.concatenate(X_train, i)
X_test = split_array[test_section]


split_array = np.array_split(y, num_sections)
for i in split_array:
      if(i != test_section):
            y_train = np.concatenate(y_train, i)
y_test = split_array[test_section]
      
# split_array = np.array_split(X, 8)
# A = split_array[0]
# B = split_array[1]
# C = split_array[2]
# D = split_array[3]
# E = split_array[4]
# F = split_array[5]
# G = split_array[6]
# H = split_array[7]
# X_train = np.concatenate((A,B,C,D,E,F,G),axis = 0)
# X_test = H

# split_array = np.array_split(y, 8)
# A = split_array[0]
# B = split_array[1]
# C = split_array[2]
# D = split_array[3]
# E = split_array[4]
# F = split_array[5]
# G = split_array[6]
# H = split_array[7]
# y_train = np.concatenate((A,B,C,D,E,F,G),axis = 0)
# y_test = H


ourSVM = SVM.SVM()

ourSVM.set(X_train)

for i in X_train:
    ourSVM.fit(X_train, y_train)

predictions = ourSVM.predict(X_test)

def accuracy(y_true, y_pred):
        total = len(y_true)
    
        accuracy = np.sum(y_true == y_pred) / total

        return accuracy


print("SVM classification accuracy", accuracy(y_test, predictions))

