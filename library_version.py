import pandas as pd
import numpy as np
import SVM
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.datasets import load_iris

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

ourSVM = SVM.SVM()
ourSVM.fit()
