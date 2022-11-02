#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 13:10:07 2019

@author: juangabriel
"""

# Regresión Lineal Múltiple

# Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el data set
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values


# Codificar datos categóricos
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import make_column_transformer

labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = make_column_transformer((OneHotEncoder(), [3]), remainder = "passthrough")
X = onehotencoder.fit_transform(X)

# Evitar la trampa de las variables ficticias
X = X[:, 1:]

# Dividir el data set en conjunto de entrenamiento y conjunto de testing CON TODO
from sklearn.model_selection import train_test_split
X_train_all, X_test_all, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Ajustar el modelo de Regresión lineal múltiple con TODO el conjunto de entrenamiento
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train_all, y_train)

# Predicción de los resultados en el conjunto de testing con TODO
y_pred_all = regression.predict(X_test_all)

# Construir el modelo óptimo de RLM utilizando la Eliminación hacia atrás

# Afegim una primera columna amb tots 1 com a requisist per la funcioó OLS

X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)


import statsmodels.api as sm

def backwardElimination(x, sl):    
    numVars = len(x[0])
    for i in range(0, numVars):   
        regressor_OLS = sm.OLS(endog = y, exog = x.tolist()).fit()        
        maxVar = max(regressor_OLS.pvalues).astype(float)        
        if maxVar > sl:            
            for j in range(0, numVars - i):                
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):                    
                    x = np.delete(x, j, 1)    
  #  regressor_OLS.summary()    
    return x 
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)

# Dividir el data set en conjunto de entrenamiento y conjunto de testing MODELAT
from sklearn.model_selection import train_test_split
X_train_mod, X_test_mod, y_train, y_test = train_test_split(X_Modeled, y, test_size = 0.2, random_state = 0)

# Ajustar el modelo de Regresión lineal múltiple MODELAT el conjunto de entrenamiento

from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train_mod, y_train)

# Predicción de los resultados en el conjunto de testing
y_pred_mod = regression.predict(X_test_mod)

"""
plt.scatter(X_test_mod[:,1:], y_pred_mod, color = "red")
plt.scatter(X_test_mod[:,1:], y_pred_all, color = "blue")
plt.scatter(X_test_mod[:,1:], y_test, color = "black")
plt.plot(X_test_mod[:,1:], y_pred_mod, color = "orange")
#plt.plot(X_test_all[:,2], y_pred_all, color = "blue")
plt.show()
"""
