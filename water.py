# Imports
import SVM
import numpy as np
import pandas as pd



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