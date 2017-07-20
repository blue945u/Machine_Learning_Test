#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
=========================================================
Linear Regression Example
=========================================================
This example uses the only the first feature of the `diabetes` dataset, in
order to illustrate a two-dimensional plot of this regression technique. The
straight line can be seen in the plot, showing how linear regression attempts
to draw a straight line that will best minimize the residual sum of squares
between the observed responses in the dataset, and the responses predicted by
the linear approximation.

The coefficients, the residual sum of squares and the variance score are also
calculated.

"""
print(__doc__)


# Code source: Jaques Grobler
# License: BSD 3 clause


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.decomposition import PCA

# Load the diabetes dataset
diabetes = pd.read_csv('data/white.csv', sep=";")  # datasets.load_diabetes()
label_column = 'quality'

# Use only one feature
diabetes_X = PCA(n_components=1).fit_transform(diabetes)

# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-200]
diabetes_X_test = diabetes_X[-200:]

# Split the targets into training/testing sets
diabetes_y_train = diabetes[label_column][:-200].values.astype(int)
diabetes_y_test = diabetes[label_column][-200:].values.astype(int)

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % np.mean((regr.predict(diabetes_X_test) - diabetes_y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(diabetes_X_test, diabetes_y_test))

print(diabetes_X_test.shape, diabetes_y_test.shape)
# Plot outputs
plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
plt.plot(diabetes_X_test, regr.predict(diabetes_X_test), color='blue',
         linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()