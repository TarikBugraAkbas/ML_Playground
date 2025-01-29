import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import kagglehub
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# download the dataset
path = kagglehub.dataset_download("govindaramsriram/energy-consumption-dataset-linear-regression")

# load the dataset
dataset_path = os.path.join(path, "train_energy_data.csv")  
dataframe = pd.read_csv(dataset_path)
test_path = os.path.join(path, "test_energy_data.csv")
test_data = pd.read_csv(test_path)

#since some of the columns are not numerical, apply one hot encoding
#dropfirst used to avoid unnecessary columns
dataframe_encoded = pd.get_dummies(dataframe, columns=['Building Type', 'Day of Week'],drop_first=True)
test_data_encoded = pd.get_dummies(test_data, columns=['Building Type', 'Day of Week'],drop_first=True)

#choose model
model = LinearRegression()

x_train = dataframe_encoded.drop(columns=['Energy Consumption'])
y_train = dataframe_encoded['Energy Consumption']

x_test = test_data_encoded.drop(columns=['Energy Consumption'])
y_test = test_data_encoded['Energy Consumption']

#train the data using x_train, y_train
model.fit(x_train,y_train)

#use x_test to predict for test.csv data
predicted = model.predict(x_test)

# evaluate performance
mse = mean_squared_error(y_test, predicted)
mae = mean_absolute_error(y_test, predicted)
r2 = r2_score(y_test, predicted)

print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("R² Score:", r2)
#Results are:
#Mean Squared Error: 0.0002015424115641789
#Mean Absolute Error: 0.012161901154076987
#R² Score: 0.9999999997063025


"""
plot the residuals (actual Energy Consumption - Predicted Using Linear Regression)
residuals = y_test - predicted

plt.figure(figsize=(10, 6))
plt.scatter(predicted, residuals, alpha=0.7, color='blue')
plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
plt.title('Residuals Plot (Actual - Predicted)', fontsize=14)
plt.xlabel('Predicted Energy Consumption', fontsize=12)
plt.ylabel('Residuals', fontsize=12)
plt.grid()
plt.show()
"""
