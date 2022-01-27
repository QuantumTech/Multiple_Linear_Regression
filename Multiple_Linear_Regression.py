# Multiple Linear Regression.

# Assumptions of linear regression:
# Ensure all these assumptions are TRUE!
# Linearity
# Homoscedasticity
# Multivariate normality
# Independence of errors
# Lack of multicollinearity

# Dummy variables:
# Find all the different categorical data within the dataset
# Turn String values into binary

# Dummy variables traps

# Pvalues! (probability value)
# Statistical significance:
# 1) Flip a coin (two possible outcomes)

# 5 Methods of building models:
# 1) All-in (step wise regression) = Use of all variables

# 2) Backward elimination (step wise regression) =
# Step 1: select a significance level to stay in the model (e.g SL = 0.05)
# Step 2: Fit the full model with all possible predictors
# Step 3: Consider the predictor with the highest P-Value. If P > SL (significance level), go to step 4, otherwise go to FIN (model is ready)
# Step 4: Remove the predictor
# Step 5: Fit model without this variable
# Return back to Step 3

# 3) Forward selection (step wise regression)
# Step 1: Select a significance level to enter the model (e.g SL = 0.05)
# Step 2: Fit all simple regression models 'Y~Xn' select the one with the lowest P-value
# Step 3: Keep this variable and fit all possible models with one extra predictor added to the one(s) you already have
# Step 4: Consider the predictor with the lowest P-value. If P<SL, go to Step 3, otherwise go to FIN

# 4) Bidirectional elimination (step wise regression)
# Step 1: Select a significance level to enter and to stay in the model
# Step 2: Perform the next step of Forward selection
# Step 3: Perform ALL steps of Backward elimination
# Step 4: No new variables can enter and no old variables can exit

# 5) Score comparison
# Step 1: Select a criterion of goodness of fit
# Step 2: Construct All possible Regression Models
# Step 3: Select the one with the best criterion

# BACKWARD ELIMINATION will be used for this project

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 1) Data preprocessing

# Importing the libraries

# 'np' is the numpy shortcut!
# 'plt' is the matplotlib shortcut!
# 'pd' is the pandas shortcut!

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 1) Data preprocessing

# Importing the dataset

# Data set is creating the data frame of the '50_Startups.csv' file
# Features (independent variables) = The columns the predict the dependent variable
# Dependent variable = The last column
# 'X' = The matrix of features (country, age, salary)
# 'Y' = Dependent variable vector (purchased (last column))
# '.iloc' = locate indexes[rows, columns]
# ':' = all rows (all range)
# ':-1' = Take all the columns except the last one
# '.values' = taking all the values

dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
#print(X)
y = dataset.iloc[:, -1].values # NOTICE! .iloc[all the rows, only the last column]
#print(y)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 1) Data preprocessing

# Encoding the Independent Variable (categorical data)

# What is happening here? = Turning the string columns (countries) into unique binary vectors

# 'One hot encoding' = Splitting a column up using the unique values. Creating binary vectors for each unique value
# 'ct' (object of the 'ColumnTransformer' class) = Creating an instance of the 'ColumnTransformer' class
# 'ColumnTransformer(transformers=[(The kind of transformation, What kind encoding, index of the columns we want to encode)], remainder = 'passthrough')'

from sklearn.compose import ColumnTransformer # Encoding 0's and 1's
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough') # 3 is the index of column we want to OneHotEcode.
X = np.array(ct.fit_transform(X))
#print(X)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 1) Data preprocessing

# Splitting the dataset into the Training set and Test set

# Note to self! = Split the data before feature scaling!
# Test set = future data
# Feature scaling = scaling the features so that they all take values in the same scale
# 80/20 split
# 'test_size' = 20% for the test set

# 'X_train' The matrix of the features of the training set
# 'X_test' The matrix of the features of the test set
# 'y_train' The dependent variable of the training set
# 'y_test' The dependent variable of the test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
#print(X_train) # The matrix of the features of the training set
#print(X_test) # The matrix of the features of the test set
#print(y_train) # The dependent variable of the training set
#print(y_test) # The dependent variable of the test set

# REMINDER MULTIPLE LINEAR REGRESSION DOES NOT NEED FEATURE SCALING!!!

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 2) Multiple Linear Regression.

# Training the Multiple Linear Regression model on the Training set
# MUlTIPLE LINEAR REGRESSION does not involve dummy variables
# The class will automatically select the most relevant model to use

from sklearn.linear_model import LinearRegression
regressor = LinearRegression() # An instance of the 'LinearRegression' class
regressor.fit(X_train, y_train) # Matrix of features, Independent variable.

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 2) Multiple Linear Regression.

# Predicting the Test set results

# '.set_printoptions(precision=2)' = Method for displaying values to a certain decimal point
# The 'len' function captures the length of the vector.
# 'np.concatenate((Vector of predicted profits (Display vertically), vector of real profits), axis (0 = means vertical, 1 = horizontal))'
# '.reshape(length of vector, number of columns)'

y_pred = regressor.predict(X_test) # Vector of dependent variables.
np.set_printoptions(precision = 2) # Displaying numerical values to 2 D.P
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1)) # Concatenate two vectors vertically (reshaping vectors)