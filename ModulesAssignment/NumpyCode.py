import numpy as np
# Create a 2D Numpy array of size 1x3 with elements of your choice
arr1=np.array([[4,5,6]])

# Create a Numpy array of length 50 with zeroes as its elements
arr2=np.zeros(50)

# Create a Numpy array of length 3x2 with elements of your choice
arr3=np.array([[2,4],[5,8],[9,10]])

#Multiply arr1 and arr3 using Numpy functions
arr4=np.dot(arr1,arr3)

# Change 5th element of arr2 to a different number
arr2[4]=100

if np.shape(arr4)==(1,2) and arr2[4]!=0:
  print("Passed")
else:
  print("Fail")