import pandas as pd

## Create a DataFrame from a dictionary
data = {
    'Name': ['Ramesh', 'Mahesh', 'Suresh'],
    'Age': [25, 30, 35],
    'City': ['Bangalore', 'Mumbai', 'Delhi']
}
df1=pd.DataFrame(data)

#Display the first 2 rows of the data frame
print(df1.head(2))

#Print the age column
print(data['Age'])

#Filter rows where age is greater than 26
filtered_df1=df1[df1['Age']>26]
print(filtered_df1)

#Add a new column 'Country' with the value 'India' for all rows
df1['Country']='India'
print(df1)

data1 = {
    'Name': ['Ramesh', 'Mahesh', 'Suresh'],
    'Age': [25, None, 35],
    'City': ['Bangalore', 'Mumbai', 'Delhi']
}

df2= pd.DataFrame(data1)

# Fill missing values in the 'Age' column with the mean age

df2.fillna({"Age":df2['Age'].mean()}, inplace=True)
print(df2)