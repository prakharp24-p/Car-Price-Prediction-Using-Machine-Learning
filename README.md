# Car-Price-Prediction-Using-Machine-Learning

Project Overview

This project develops a machine learning model for predicting the selling price of used cars based on vehicle characteristics such as age, fuel type, transmission type, ownership history, and mileage.

The objective is to build an accurate regression model capable of estimating a car's market value using historical sales data. Multiple regression techniques are evaluated and compared to identify the best-performing model.

The project includes:

• Data preprocessing and cleaning • Feature engineering • Categorical variable encoding • Model training and evaluation • Hyperparameter tuning • Feature importance analysis • Model persistence using Pickle

Dataset Description

The dataset contains information about used cars and their selling prices.

Features include:

• Year • Present Price • Kms Driven • Fuel Type • Seller Type • Transmission • Owner

Target Variable:

• Selling_Price

Project Workflow

Data Loading
Data Cleaning
Feature Engineering
Data Encoding
Train-Test Split
Model Training
Hyperparameter Optimization
Model Evaluation
Feature Importance Analysis
Model Saving
Data Preprocessing

The following preprocessing steps were performed:

• Removed the Car_Name column. • Created a Car_Age feature using the manufacturing year. • Removed the original Year column. • Checked for missing values. • Applied One-Hot Encoding to categorical features.

Categorical Variables

• Fuel_Type • Seller_Type • Transmission

Feature Engineering

A new feature was created:

Car_Age = Current Year − Manufacturing Year

This feature better captures vehicle depreciation and improves predictive performance.

Machine Learning Models

Linear Regression

Linear Regression serves as the baseline model and captures linear relationships between vehicle attributes and selling price.

Lasso Regression

Lasso Regression introduces L1 regularization to reduce overfitting and automatically shrink less important feature coefficients.

Hyperparameter tuning was performed using GridSearchCV to identify the optimal alpha value.

Model Evaluation

Models were evaluated using:

• R² Score • Mean Squared Error (MSE)

Higher R² values and lower MSE values indicate better performance.

Overfitting Analysis

Training and testing R² scores were compared to detect potential overfitting.

If:

Train R² − Test R² > 0.10

the model may be overfitting.

Feature Importance Analysis

Feature coefficients from the Linear Regression model were analyzed to identify the variables that most strongly influence vehicle prices.

Visualizations

The project generates:

• Actual vs Predicted Price plot for Linear Regression • Actual vs Predicted Price plot for Lasso Regression

These visualizations help assess prediction quality and identify systematic errors.

Model Selection

Both models are compared using:

• Test R² Score • Mean Squared Error

The best-performing model is selected automatically.

Model Persistence

The final model is saved as:

car_price_model.pkl

The saved model can be loaded later without retraining.

Example:

import pickle

model = pickle.load(open('car_price_model.pkl', 'rb'))

Project Structure

project/

├── car_price_pred.py ├── car data.csv ├── car_price_model.pkl ├── README.md └── requirements.txt

Required Libraries

• pandas • numpy • matplotlib • seaborn • scikit-learn

Install dependencies:

pip install pandas numpy matplotlib seaborn scikit-learn

How to Run

Place the dataset file (car data.csv) in the project directory.

Run:

python car_price_pred.py

View:
• Performance metrics • Model comparison table • Feature importance analysis • Prediction plots

Access the saved model:
car_price_model.pkl

Key Features

• Data preprocessing pipeline • One-Hot Encoding • Feature engineering • Linear Regression • Lasso Regression • Hyperparameter tuning • Model comparison • Overfitting detection • Feature importance analysis • Model persistence

Future Improvements

Possible extensions include:

• Random Forest Regressor • XGBoost Regressor • Gradient Boosting • Cross-validation studies • Flask/FastAPI deployment • Interactive prediction dashboard

Conclusion

This project demonstrates an end-to-end machine learning workflow for car price prediction. Through feature engineering, robust preprocessing, regression modeling, hyperparameter optimization, and model comparison, the system provides accurate and interpretable predictions for used vehicle valuation.
