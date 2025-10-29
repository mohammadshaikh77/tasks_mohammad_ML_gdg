GDG Task 1: Crab Data Analysis
This project focuses on performing Exploratory Data Analysis (EDA) on a dataset containing information about crabs. The goal is to understand the characteristics of the data, identify patterns, and prepare it for potential machine learning tasks.

Dataset
The dataset used in this project is related to crabs and can be downloaded from: [redacted link]

Project Steps
The following steps have been performed in this project:

Library Installation and Import: Essential libraries such as ydata-profiling, numpy, pandas, matplotlib.pyplot, and seaborn were installed and imported.
Google Drive Mounting: Google Drive was mounted to access the dataset file.
Data Loading: The crab dataset was loaded into a pandas DataFrame.
Basic EDA:
The first few rows of the DataFrame were viewed to inspect the data.
Column names were listed.
The 'id' column was dropped as it was not needed for analysis.
The shape (number of rows and columns) of the DataFrame was checked.
The info() function was used to get a summary of the DataFrame, including data types and non-null counts.
The describe() function was used to view descriptive statistics of the numerical columns.
Unique values in the 'Sex' and 'Age' columns were printed.
The count of crabs for each gender was determined.
Pandas Profiling: The ydata-profiling library was used to generate a comprehensive profile report of the dataset, providing an automated overview of the data.
Data Cleaning: Rows where the 'Height' was zero were removed from the dataset.
Data Visualization:
A bar chart was plotted to visualize the average age of each sex.
A boxplot was created to show the distribution of age by sex.
KDE plots were generated for 'Age', 'Length', and 'Weight' to visualize their distributions.
A scatterplot was created to show the relationship between 'Age' and 'Diameter'.
A pairplot was generated to visualize relationships between multiple numeric features, colored by the 'Lost Weight' category.
A heatmap was created to visualize the correlation matrix of the numeric features.
Feature Engineering: A new column 'Lost Weight' was created based on the difference between 'Weight' and the sum of 'Shucked Weight', 'Viscera Weight', and 'Shell Weight'. This column was then converted into a binary (0 or 1) representation.
Feature Scaling and Encoding:
One-hot encoding was applied to the 'Sex' column to convert categorical data into a numerical format suitable for machine learning algorithms.
The dataset was normalized using MinMaxScaler, excluding the one-hot encoded columns.
The dataset was standardized using StandardScaler, excluding the one-hot encoded columns. The boolean values in the one-hot encoded columns were converted to integers (0 and 1).
Libraries Used
pandas
numpy
matplotlib.pyplot
seaborn
ydata-profiling
sklearn.preprocessing.MinMaxScaler
sklearn.preprocessing.StandardScaler
sklearn.preprocessing.OneHotEncoder
Insights Gained
The dataset is relatively balanced in terms of sex distribution, with a slight skew towards male crabs.
Female crabs appear to have a slightly longer average lifespan compared to male and infant crabs, based on the average age bar chart.
Boxplots provide a more detailed view of the age distribution across different sexes, showing variations in quartiles and potential outliers.
KDE plots show the density distribution of numerical features.
Scatterplots and pairplots help visualize the relationships and correlations between different features.
The correlation heatmap reveals strong positive correlations between various weight and size measurements, and a moderate correlation between Shell Weight and Age.
The 'Lost Weight' feature was engineered to capture potential missing weight components and its relationship with age was explored using a violin plot.
One-hot encoding, normalization, and standardization were applied to prepare the data for machine learning models.
Future Work
Explore more advanced data visualization techniques.
Perform further feature engineering based on domain knowledge or insights from EDA.
Build and evaluate machine learning models to predict crab age or other relevant outcomes.
Investigate outliers and their impact on the analysis.
