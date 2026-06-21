
# IMPORT LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# LOAD DATA

car_data = pd.read_csv("car data.csv")

print("Dataset Shape:", car_data.shape)


# DATA PREPROCESSING


# Remove car name since it doesn't add much predictive power
car_data.drop("Car_Name", axis=1, inplace=True)

# Create Car Age Feature
CURRENT_YEAR = 2025
car_data["Car_Age"] = CURRENT_YEAR - car_data["Year"]

# Drop Year column
car_data.drop("Year", axis=1, inplace=True)

# Check missing values
print("\nMissing Values:")
print(car_data.isnull().sum())


# FEATURES AND TARGET

X = car_data.drop("Selling_Price", axis=1)
y = car_data["Selling_Price"]

# Identify categorical columns
categorical_cols = [
    "Fuel_Type",
    "Seller_Type",
    "Transmission"
]

# One-Hot Encoding
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first"), categorical_cols)
    ],
    remainder="passthrough"
)


# TRAIN TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.1,
    random_state=42
)


# LINEAR REGRESSION

linear_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])

linear_pipeline.fit(X_train, y_train)

train_pred_lr = linear_pipeline.predict(X_train)
test_pred_lr = linear_pipeline.predict(X_test)

lr_train_r2 = r2_score(y_train, train_pred_lr)
lr_test_r2 = r2_score(y_test, test_pred_lr)

lr_mse = mean_squared_error(y_test, test_pred_lr)


# LASSO REGRESSION

lasso_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", Lasso())
])

param_grid = {
    "model__alpha": [0.001, 0.01, 0.1, 1, 10]
}

grid_search = GridSearchCV(
    lasso_pipeline,
    param_grid,
    cv=5,
    scoring="r2"
)

grid_search.fit(X_train, y_train)

best_lasso = grid_search.best_estimator_

train_pred_lasso = best_lasso.predict(X_train)
test_pred_lasso = best_lasso.predict(X_test)

lasso_train_r2 = r2_score(y_train, train_pred_lasso)
lasso_test_r2 = r2_score(y_test, test_pred_lasso)

lasso_mse = mean_squared_error(y_test, test_pred_lasso)


# MODEL COMPARISON

results = pd.DataFrame({
    "Model": ["Linear Regression", "Lasso Regression"],
    "Train R2": [lr_train_r2, lasso_train_r2],
    "Test R2": [lr_test_r2, lasso_test_r2],
    "MSE": [lr_mse, lasso_mse]
})

print("\nModel Comparison")
print(results)


# OVERFITTING CHECK

for model, train_r2, test_r2 in zip(
        results["Model"],
        results["Train R2"],
        results["Test R2"]):

    gap = train_r2 - test_r2

    print(f"\n{model}")

    if gap > 0.10:
        print("Potential Overfitting")
    else:
        print("No Significant Overfitting")

# FEATURE IMPORTANCE

feature_names = (
    linear_pipeline.named_steps["preprocessor"]
    .get_feature_names_out()
)

coefficients = (
    linear_pipeline.named_steps["model"]
    .coef_
)

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coefficients
})

importance_df["Abs_Coefficient"] = abs(
    importance_df["Coefficient"]
)

importance_df = importance_df.sort_values(
    "Abs_Coefficient",
    ascending=False
)

print("\nTop Features")
print(importance_df.head(10))


# VISUALIZATION


plt.figure(figsize=(8,6))
plt.scatter(y_test, test_pred_lr)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Linear Regression Predictions")
plt.show()

plt.figure(figsize=(8,6))
plt.scatter(y_test, test_pred_lasso)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Lasso Regression Predictions")
plt.show()


# SAVE BEST MODEL

if lr_test_r2 >= lasso_test_r2:
    best_model = linear_pipeline
    model_name = "Linear Regression"
else:
    best_model = best_lasso
    model_name = "Lasso Regression"

pickle.dump(
    best_model,
    open("car_price_model.pkl", "wb")
)

print("\nBest Model:", model_name)
print("Model saved as car_price_model.pkl")
