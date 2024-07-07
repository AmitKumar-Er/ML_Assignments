import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Aim is to predict the marks of students of the test data
# Use the file named 'training data' to train the model
data = pd.read_excel("C:\\Users\\ak248\\ML_LS_W1\\ML_Assignments\\Linear_Regression_Assignment\\Training data.xlsx")
x_train = np.array(data.iloc[:,0:8])
y_train = np.array(data.iloc[:,8]).reshape(-1,1)

#plotting Y_train with different features
feature_names = data.columns[:8]
for i in range(x_train.shape[1]):
    plt.figure(figsize=(8, 6))
    plt.scatter(x_train[:, i], y_train, alpha=0.5)
    plt.xlabel(f'Feature {i+1}')
    plt.ylabel('Target')
    plt.title(f'Feature {i+1} vs Target')
    plt.show()
    

def feature_changing(x_train):
    #replacing yes with numerical value 1 and no with 0
    #replacing M with 1 and F with 0
      
    x_train[:, 0] = np.where(x_train[:, 0] == 'yes', 1, np.where(x_train[:, 0] == 'no', 0, x_train[:, 0]))
    x_train[:, 1] = np.where(x_train[:, 1] == 'M', 1, np.where(x_train[:, 1] == 'F', 0, x_train[:, 1]))
    
    return x_train

x_train = feature_changing(x_train)
print(x_train)


def z_score(x_train):
    
     # Compute the mean and standard deviation for each feature
     x_mean = np.mean(x_train, axis=0)
     x_std = np.std(x_train, axis=0, dtype=float)
    
     # Perform the z-score standardization
     x_train = (x_train - x_mean) / x_std
    
     return x_train, x_std, x_mean

# # Call the z_score function
x_train_standardized, x_std, x_mean = z_score(x_train)

# Print the results
print("Standardized x_train:\n", x_train_standardized)
print("Standard Deviations:\n", x_std)
print("Means:\n", x_mean)
print(x_train)