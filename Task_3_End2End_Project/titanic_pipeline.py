# titanic_pipeline.py

import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# Load dataset from seaborn
df = sns.load_dataset("titanic")

# Drop irrelevant columns
df.drop(["deck", "embark_town", "alive", "class", "who", "adult_male"], axis=1, inplace=True)

# Drop rows with too many nulls
df.dropna(subset=["embarked", "age", "embarked"], inplace=True)

# Fill missing values
df["age"].fillna(df["age"].median(), inplace=True)
df["embarked"].fillna(df["embarked"].mode()[0], inplace=True)

# Encode categorical columns
df["sex"] = df["sex"].map({"male": 0, "female": 1})
df["embarked"] = df["embarked"].map({"S": 0, "C": 1, "Q": 2})
df["alone"] = df["alone"].astype(int)

# Define features and target
X = df[["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked", "alone"]]
y = df["survived"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/titanic_model.pkl")
print("Model saved to models/titanic_model.pkl")
