# Data handling
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# ML tools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

#Load Dataset
df = pd.read_csv("StudentsPerformance.csv")

print("First 5 Rows:")
print(df.head())

#Data Checking
print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Values:")
print(df.duplicated().sum())

print("\nStatistical Summary:")
print(df.describe())

#Exploratory Data Analysis
#Gender vs Math Score
plt.figure()
sns.boxplot(x='gender', y='math score', data=df)
plt.title("Gender vs Math Score")
plt.show()

#Test Preparation vs Math Score
plt.figure()
sns.boxplot(x='test preparation course', y='math score', data=df)
plt.title("Test Preparation vs Math Score")
plt.show()

#Correlation Heatmap

plt.figure()
sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.title("Correlation Matrix")
plt.show()

#Define Features & Target
X = df.drop("math score", axis=1)
y = df["math score"]

#Encoding
categorical_cols = X.select_dtypes(include=["object", "string"]).columns
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first"), categorical_cols)
    ],
    remainder="passthrough"
)

#Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model 1 – Linear Regression
lr_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])

lr_pipeline.fit(X_train, y_train)
lr_pred = lr_pipeline.predict(X_test)

print("\nLinear Regression Results")
print("R2 Score:", r2_score(y_test, lr_pred))
print("MAE:", mean_absolute_error(y_test, lr_pred))


#Model 2 – Random Forest
rf_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(random_state=42))
])

rf_pipeline.fit(X_train, y_train)
rf_pred = rf_pipeline.predict(X_test)

print("\nRandom Forest Results")
print("R2 Score:", r2_score(y_test, rf_pred))
print("MAE:", mean_absolute_error(y_test, rf_pred))

#Model Comparison Graph
lr_r2 = r2_score(y_test, lr_pred)
rf_r2 = r2_score(y_test, rf_pred)

models = ["Linear Regression", "Random Forest"]
scores = [lr_r2, rf_r2]

plt.figure()
plt.bar(models, scores)
plt.xlabel("Models")
plt.ylabel("R2 Score")
plt.title("Model Comparison")
plt.show()

#Actual vs Predicted Graph

plt.figure()
plt.scatter(y_test, rf_pred)
plt.xlabel("Actual Math Score")
plt.ylabel("Predicted Math Score")
plt.title("Actual vs Predicted (Random Forest)")
plt.show()

