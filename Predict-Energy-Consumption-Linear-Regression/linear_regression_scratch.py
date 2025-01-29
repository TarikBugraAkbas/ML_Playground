import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import kagglehub
import os


path = kagglehub.dataset_download("govindaramsriram/energy-consumption-dataset-linear-regression")

# Load the dataset
dataset_path = os.path.join(path, "train_energy_data.csv")  
dataframe = pd.read_csv(dataset_path)
test_path = os.path.join(path, "test_energy_data.csv")
test_data = pd.read_csv(test_path)

# Apply one-hot encoding to non-numerical columns
dataframe_encoded = pd.get_dummies(dataframe, columns=['Building Type', 'Day of Week'], drop_first=True)
test_data_encoded = pd.get_dummies(test_data, columns=['Building Type', 'Day of Week'], drop_first=True)

# Extract features (X) and target (y)
x_train = dataframe_encoded.drop(columns=['Energy Consumption'])
y_train = dataframe_encoded['Energy Consumption']

x_test = test_data_encoded.drop(columns=['Energy Consumption'])
y_test = test_data_encoded['Energy Consumption']


"""
for j in x_train.columns:
    x_train[j] = (x_train[j] - x_train[j].mean()) / x_train[j].std()
"""
#apply feature scaling
train_mean = x_train.mean()
train_std = x_train.std()

x_train = (x_train - train_mean) / train_std
x_test = (x_test - train_mean) / train_std


#print(x_test.isnull().sum())

def cost_function(x, y, w, b):
    size = x.shape[0]
    f = np.dot(x, w) + b  # vectorized predictions
    cost = (1 / (2 * size)) * np.sum((f - y) ** 2)  # Mean Squared Error
    return cost

def gradient_descent(x, y, w, b, alpha, iteration_count):
    cost_list = []
    i= 0
    epsilon = 0.00000001 #convergence threshold
    while(i< iteration_count):    
        size = x.shape[0]
        f = np.dot(x, w) + b 
        db = (1 / size) * np.sum(f - y)  
        dw = (1 / size) * np.dot(x.T, (f - y))

        # update weights and bias
        w -= alpha * dw
        b -= alpha * db
        cost = cost_function(x,y,w,b)
        cost_list.append(cost)
        
        #if found minimum, do not continue
        if(i > 0 and (abs(cost_list[-1] - cost_list[-2]) < epsilon)):
            #print("converged at i = ",i)
            break
        
        i += 1
    return w, b, cost_list, i

#initialize w and b
w = np.zeros(x_train.shape[1])  #feature count sized array
b = 0 
alpha = 0.1 # learning rate
iteration_count = 1000

#run gradient descent
w,b,cost_list,i = gradient_descent(x_train.to_numpy(), y_train.to_numpy(), w, b, alpha, iteration_count)

predictions = np.dot(x_test, w) + b

# evaluate performance

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)

print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("R² Score:", r2)
#Results are:
#Mean Squared Error: 0.00020172790190814134
#Mean Absolute Error: 0.012201787145131674
#R² Score: 0.9999999997060321


"""
plot the residuals (actual Energy Consumption - Predicted Using Linear Regression)
residuals = y_test - predictions

plt.figure(figsize=(10, 6))
plt.scatter(predictions, residuals, alpha=0.7, color='blue')
plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
plt.title('Residuals Plot (Actual - Predicted)', fontsize=14)
plt.xlabel('Predicted Energy Consumption', fontsize=12)
plt.ylabel('Residuals', fontsize=12)
plt.grid()
plt.show()
"""