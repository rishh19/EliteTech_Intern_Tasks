"""
Simple ETL Pipeline using pandas and scikit-learn
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Step 1: Create some sample data
data = {
    'Name': ['Ravi', 'Priya', 'Amit', 'Neha', 'Karan'],
    'Age': [24, np.nan, 35, 42, 29],
    'Gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
    'Salary': [40000, 50000, np.nan, 75000, 60000],
    'Department': ['Sales', 'HR', 'HR', 'Marketing', 'Sales']
}

df = pd.DataFrame(data)
print("Original Data:")
print(df)

# Step 2: Fill missing values with mean
df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Salary'].fillna(df['Salary'].mean(), inplace=True)

# Step 3: Apply transformations
numeric_cols = ['Age', 'Salary']
categorical_cols = ['Gender', 'Department']

num_transformer = StandardScaler()
cat_transformer = OneHotEncoder(drop='first')

preprocessor = ColumnTransformer([
    ('num', num_transformer, numeric_cols),
    ('cat', cat_transformer, categorical_cols)
])

pipeline = Pipeline([('preprocessor', preprocessor)])
transformed = pipeline.fit_transform(df)

# Get final column names
encoded_names = pipeline.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out(categorical_cols)
final_columns = ['Age_scaled', 'Salary_scaled'] + list(encoded_names)
final_df = pd.DataFrame(transformed, columns=final_columns)

print("\nFinal Transformed Data:")
print(final_df)

# Step 4: Save the data
final_df.to_csv('processed_data.csv', index=False)
print("\nSaved to 'processed_data.csv'")
