# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_excel("PPM output.xlsx")

# Select important columns
df.columns

dfModel = df[['WO','Line','Status 2']]

# Create dummy Data
dfDummy = pd.get_dummies(dfModel)

# Train test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# multiple linear regression
import statsmodels.api as sm

X_sm = X = sm.add_constant(X)
model = sm.OLS(y, X_sm)
model.fit().summary()

# Other regrassion model
# test ensembles