import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn import svm
import pandas as pd
import ntpath
import random
import seaborn as sns

columns = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity', 'Potability']
data = pd.read_csv('water_potability.csv', names=columns, skiprows=1)

pd.set_option('display.max_colwidth', None)

def filtering(arr):
    temp = np.isnan(arr)
    if True in temp:
        return False
    else:
        return True

print(data.head())
samples = len(data)
y = data["Potability"]
data = data.drop("Potability", axis=1)

x = data.to_numpy()
filtered_x = filter(filtering, x)

x = []
for i in filtered_x:
    x.append(i)

# model = svm.SVC()
# model.fit(filtered_x,y)

print(x[0])