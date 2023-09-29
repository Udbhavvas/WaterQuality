# Imports
import SVM
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


# Obtaining/Cleaning CSV file
columns = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity', 'Potability']
data = pd.read_csv('water_potability.csv', names=columns, skiprows=1)

# Drop rows with NaN values
data = data.dropna()

# Randomizes data points
data = data.sample(frac=1)

# Drops potability column from the data read in from csv
pot = data["Potability"]
data = data.drop("Potability", axis=1)

# Instatiate SVM object
SVM_instance = SVM.SVM()

#Places data minus potability into X and potability data into Y
X = np.array(data)
y = np.array(pot)

# Reassign labels as -1 and 1
y = np.where(y == 0, -1, 1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=123
)

SVM_instance.train(X_train, y_train)

predictions = SVM_instance.predict(X_test)

print("SVM classification accuracy", SVM_instance.accuracy(y_test, predictions))

