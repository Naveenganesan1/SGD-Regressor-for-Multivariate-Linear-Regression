# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load California housing data, select three features as X and two target variables as Y, then split into train and test sets.
   
2. Standardize X and Y using StandardScaler for consistent scaling across features.

3. Initialize SGDRegressor and wrap it with MultiOutputRegressor to handle multiple targets.

4. Train the model on the standardized training data.

5. Predict on the test data, inverse-transform predictions, compute mean squared error, and print results.
   
## Program:

Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.

# Developed by: NAVEEN G
# RegisterNumber: 212223220066
```
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

#Load the California Housing dataset
dataset = fetch_california_housing()
df=pd.DataFrame(dataset.data,columns=dataset.feature_names)
df['HousingPrice']=dataset.target
print(df.head())

#Use the first 3 feature as inputs
X = df.drop(columns=['AveOccup','HousingPrice'])
Y = df[['AveOccup','HousingPrice']]

#Split the data into training and testing sets
X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)

#Scale the feature and target variables
scaler_X = StandardScaler()
scaler_Y = StandardScaler()

X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)
Y_train = scaler_Y.fit_transform(Y_train)
Y_test = scaler_Y.transform(Y_test)

#Initialize the SGDRegressor
sgd = SGDRegressor(max_iter=1000,tol=1e-3)

multi_output_sgd = MultiOutputRegressor(sgd)

multi_output_sgd.fit(X_train,Y_train)

Y_pred = multi_output_sgd.predict(X_test)

Y_pred = scaler_Y.inverse_transform(Y_pred)
Y_test = scaler_Y.inverse_transform(Y_test)

mse= mean_squared_error(Y_test,Y_pred)
print("Mean Squared Error:",mse)

print("\nPredictions:\n",Y_pred[:5])


```

## Output:

![Screenshot 2025-03-10 155822](https://github.com/user-attachments/assets/c28f7e1b-c900-433c-8dbe-e1ee35cb1fec)

## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
