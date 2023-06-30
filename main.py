from sklearn.datasets import fetch_california_housing
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV


# Data Collection

data = fetch_california_housing()

df = pd.DataFrame(data.data, columns=data.feature_names)
print(df.describe())

# Data Exploration

df.hist(bins=30, figsize=(20, 15))
plt.show()

# Data Preprocessing

imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(df)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, data.target, test_size=0.2, random_state=42)

# Model Selection and Training

model = RandomForestRegressor(random_state=420)
model.fit(X_train, y_train)

# Model Evaluation

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Test set:")

print("Mean squared error:", mse)
print("Root Mean Squared Error:", rmse)
print("R-squared Score:", r2)

cv_scores = cross_val_score(model, X_scaled, data.target, cv=5, scoring='neg_mean_squared_error')

mse_cv = -np.mean(cv_scores)
rmse_cv = np.sqrt(mse_cv)
r2_cv = np.mean(cross_val_score(model, X_scaled, data.target, cv=5, scoring='r2'))

print("Cross-Validation:")

print("Mean Squared Error:", mse_cv)
print("Root Mean Squared Error:", rmse_cv)
print("R-squared Score:", r2_cv)

# Model Turning

param_grid = {'random_state': [42, 420, 4200]}

grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_

print("Best Params:", best_params)

# Results Visualization

plt.scatter(y_test, y_pred)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted House Prices')
plt.show()

importance = model.feature_importances_
feature_names = data.feature_names

plt.barh(feature_names, importance)
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Feature Importance')
plt.show()
