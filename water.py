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

ourSVM = SVM.SVM()

ourSVM.set_weights(X)

num_sections = 8

X_split_array = np.array_split(X, num_sections)
y_split_array = np.array_split(y, num_sections)


j = 0
while j < num_sections:
      test_section = j

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

      j += 1
      




print("SVM classification accuracy", ourSVM.accuracy(y_test, predictions))

