import matplotlib.pyplot as plt
import numpy as np

xpoints=np.array([1,2,3,4])
ypoints=np.array([2,4,6,8])

#Plot these points without drawing a line
plt.plot(xpoints,ypoints,linestyle="None",marker='o')
plt.show()

#Plotting with marker: Plot these points with a marker(Star marker)
plt.plot(xpoints,ypoints,linestyle="None",marker='*')
plt.show()

#Using fmt format, add circular marker,red color and Dashed line
plt.plot(xpoints,ypoints,'ro--')
plt.show()

#Add xlabel,ylabel and title for the plot.
plt.plot(xpoints,ypoints,'ro--')
plt.xlabel("xpoints")
plt.ylabel("ypoints")
plt.title("x-y points plotting")
plt.show()

#Create a scatter plot for xpoints and ypoints
plt.scatter(xpoints,ypoints)
plt.show()

#Set color to the scatter plot. Blue,Green,Red and yellow color for each point respectively
colors = ['blue', 'green', 'red', 'yellow']
plt.scatter(xpoints, ypoints, c=colors)
plt.xlabel('x-points')
plt.ylabel('y-points')
plt.title('Scatter Plot')
plt.show()