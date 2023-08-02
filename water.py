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

split_array = np.array_split(X, 4)
A = split_array[0]
B = split_array[1]
C = split_array[2]
D = split_array[3]
X_train = np.concatenate((A,B,D),axis = 0)
X_test = C

split_array = np.array_split(y, 4)
A = split_array[0]
B = split_array[1]
C = split_array[2]
D = split_array[3]
y_train = np.concatenate((A,B,D),axis = 0)
y_test = C


ourSVM = SVM.SVM()

ourSVM.fit(X_train, y_train)

predictions = ourSVM.predict(X_test)

def accuracy(y_true, y_pred):
        total = len(y_true)
    
        accuracy = np.sum(y_true == y_pred) / total

        return accuracy


print("SVM classification accuracy", accuracy(y_test, predictions))

