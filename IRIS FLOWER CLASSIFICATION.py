# Step 1 – Load the data:
# DataFlair Iris Flower Classification
# Import Packages
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
%matplotlib inline
# First, we’ve imported some necessary packages for the project.

# Numpy will be used for any computational operations.
# We’ll use Matplotlib and seaborn for data visualization.
# Pandas help to load data from various sources like local storage, database, excel file, CSV file, etc.
 columns = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width', 'Class_labels'] 
# Load the data
df = pd.read_csv('iris.data', names=columns)
df.head()
# Next, we load the data using pd.read_csv() and set the column name as per the iris data information.
# Pd.read_csv reads CSV files. CSV stands for comma separated value.
# df.head() only shows the first 5 rows from the data set table.
# All the numerical values are in centimeters.
# Step 2 – Analyze and visualize the dataset:
# Let’s see some information about the dataset.
# Some basic statistical analysis about the data
df.describe()
# From this description, we can see all the descriptions about the data, like average length and width, minimum value, maximum value, the 25%, 50%, and 75% distribution value, etc.

# Let’s visualize the dataset.
