# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import title
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LinearRegression

# Load the housing dataset
housing = pd.read_csv('housing.csv')

# Step 1: Create an 'income category' column for stratified sampling
housing['income_cat'] = pd.cut(housing['median_income'],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])

# Step 2: Stratified split into training and testing sets
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# Remove the 'income_cat' column from training and test sets
for set_ in (strat_train_set, strat_test_set):
    set_.drop('income_cat', axis=1, inplace=True)

# Step 3: Visualization of data distribution (scatter plot)
housing = strat_train_set.copy()
housing.plot(kind='scatter', x='longitude', y='latitude',title = 'heatmap California', alpha=0.4,
             s=housing['population'] / 100, label='Population', figsize=(12, 8),
             c='median_house_value', cmap=plt.get_cmap('jet'), colorbar=True)
plt.legend()
plt.show()

# Step 4: Prepare data for Linear Regression
housing = strat_train_set.drop("median_house_value", axis=1)  # Features
housing_labels = strat_train_set["median_house_value"].copy()  # Target

# One-hot encode 'ocean_proximity' and drop the first category to avoid collinearity
housing = pd.get_dummies(housing, columns=['ocean_proximity'], drop_first=True)

# Step 5: Handle missing values in 'total_bedrooms' by filling with the median
housing.fillna({'total_bedrooms': housing['total_bedrooms'].median()}, inplace=True)

# Step 6: Train the Linear Regression model
lin_reg = LinearRegression()
lin_reg.fit(housing, housing_labels)

# Step 7: Make predictions on a sample of the training set
sample_data = housing.iloc[:5]
sample_labels = housing_labels.iloc[:5]
predictions = lin_reg.predict(sample_data)

# Output the predictions and actual values for comparison
print("Predictions:", predictions)
print("Actual values:", sample_labels.values)
