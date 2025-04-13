# Importing necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Task 1: Load and Explore the Dataset
# Load the Iris dataset from seaborn (this can be swapped with pd.read_csv() for other datasets)
data = sns.load_dataset('iris')

# Display the first few rows
print("First 5 rows of the dataset:")
print(data.head())

# Check the structure of the dataset
print("\nDataset Structure:")
print(data.info())

# Check for missing values
print("\nMissing Values in the Dataset:")
print(data.isnull().sum())

# If there were missing values, you could drop or fill them like this:
# data = data.dropna()  # To drop rows with missing values
# data = data.fillna(value)  # To fill missing values with a specific value

# Task 2: Basic Data Analysis
# Compute basic statistics of numerical columns
print("\nBasic Statistics (mean, median, std, etc.):")
print(data.describe())

# Perform groupings and compute the mean of a numerical column by 'species'
grouped_data = data.groupby('species')['sepal_length'].mean()
print("\nAverage Sepal Length per Species:")
print(grouped_data)

# Task 3: Data Visualization
# 1. Line chart showing trends (using 'index' for simulation)
data['index'] = range(1, len(data) + 1)  # Simulating a trend with an index column
plt.figure(figsize=(10, 6))
plt.plot(data['index'], data['sepal_length'], label='Sepal Length')
plt.title('Sepal Length Trend')
plt.xlabel('Index')
plt.ylabel('Sepal Length (cm)')
plt.legend()
plt.show()

# 2. Bar chart: Average petal length per species
plt.figure(figsize=(10, 6))
sns.barplot(x='species', y='petal_length', data=data)
plt.title('Average Petal Length per Species')
plt.xlabel('Species')
plt.ylabel('Average Petal Length (cm)')
plt.show()

# 3. Histogram of sepal length
plt.figure(figsize=(10, 6))
sns.histplot(data['sepal_length'], kde=True, bins=15)
plt.title('Distribution of Sepal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Frequency')
plt.show()

# 4. Scatter plot: Sepal length vs Petal length
plt.figure(figsize=(10, 6))
sns.scatterplot(x='sepal_length', y='petal_length', hue='species', data=data)
plt.title('Sepal Length vs Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend(title='Species')
plt.show()

# Task 4: Findings and Observations
# Based on the analysis and visualizations, summarize your findings.

# Example Observations:
# 1. The sepal length differs among species, with **setosa** having the shortest average sepal length.
# 2. Petal length and sepal length have a positive correlation, especially for **versicolor** and **virginica**.
# 3. The histogram shows a normal distribution of sepal lengths across species, indicating that it is not skewed.
