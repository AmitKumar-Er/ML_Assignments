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



def z_score(x_train):
    
     # Compute the mean and standard deviation for each feature
     x_mean = np.mean(x_train, axis=0)
     x_std = np.std(x_train, axis=0, dtype=float)
    
     # Perform the z-score standardization
     x_train = (x_train - x_mean) / x_std
    
     return x_train, x_std, x_mean


def cost(x_train,y_train,w,b):

    # Use mean square error as cost function
    # return cost

    # Number of training examples
    m = x_train.shape[0]
    
    # Compute the predicted values
    y_pred = np.dot(x_train, w) + b
    
    # Compute the Mean Squared Error
    loss = (1 / (2 * m)) * np.sum((y_pred - y_train) ** 2)

    return loss
np.random.seed(2147483647)
w = np.random.randn(x_train.shape[1],1)
b = np.random.randn(1)
p=cost(x_train,y_train,w,b)
print(p)

def gradient_descent(x_train,y_train,w,b):
    
    # Number of training examples
    m = x_train.shape[0]
    
    # Fixed learning rate and number of iterations
    learning_rate = 1
    num_iterations = 1000
    
    for i in range(num_iterations):
        # Compute the predicted values
        y_pred = np.dot(x_train, w) + b
        
        # Compute the gradients
        dw = (1 / m) * np.dot(x_train.T, (y_pred - y_train))
        db = (1 / m) * np.sum(y_pred - y_train)
        
        # Update the weights and bias
        w -= learning_rate * dw
        b -= learning_rate * db
        
        # Optionally, print the cost every 100 iterations for monitoring
        if i % 100 == 0:
            cost = (1 / (2 * m)) * np.sum((y_pred - y_train) ** 2)
            print(f"Iteration {i}: Cost {cost}")
            
    return w,b

