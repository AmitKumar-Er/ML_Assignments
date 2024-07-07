import numpy as np

# Create the 3x3 identity matrix
I = np.eye(3)

# Calculate 9I + 1 as P
P= 9 * I + 1

# Perform the dot product of I and (9I + 1)
result = np.dot(I,P)
#showing the result
print(result)
