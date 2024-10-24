import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
file_path = "city_day.csv"
df = pd.read_csv(file_path)

# Data Cleaning: Handle missing values using KNN Imputer
imputer = KNNImputer(n_neighbors=5)
numeric_columns = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'AQI']
df[numeric_columns] = imputer.fit_transform(df[numeric_columns])

# Data Cleaning: Remove outliers using Interquartile Range (IQR) method
Q1 = df[numeric_columns].quantile(0.25)
Q3 = df[numeric_columns].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df[numeric_columns] < (Q1 - 1.5 * IQR)) | (df[numeric_columns] > (Q3 + 1.5 * IQR))).any(axis=1)]

# Feature Engineering: Adding new features such as 'Season' and pollutant ratios
df['Season'] = pd.to_datetime(df['Date']).dt.month % 12 // 3 + 1  # Deriving Season from the Date column
df['PM2.5/PM10'] = df['PM2.5'] / df['PM10']

# Data Normalization: Apply Min-Max Scaling
scaler = MinMaxScaler()
scaled_columns = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'PM2.5/PM10']
df[scaled_columns] = scaler.fit_transform(df[scaled_columns])

# Define features and target
X = df[['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'Season', 'PM2.5/PM10']]
y = df['AQI']

# Split the data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Initialize models
linear_model = LinearRegression()
random_forest_model = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42)
svr_model = SVR(kernel='rbf', C=1.0, gamma='scale')

# Train models
linear_model.fit(X_train, y_train)
random_forest_model.fit(X_train, y_train)
svr_model.fit(X_train, y_train)

# Validate models
def evaluate_model(model, X_val, y_val):
    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_val, y_pred)
    return mse, rmse, r2

# Evaluate Linear Regression
mse_lr, rmse_lr, r2_lr = evaluate_model(linear_model, X_val, y_val)
# Evaluate Random Forest
mse_rf, rmse_rf, r2_rf = evaluate_model(random_forest_model, X_val, y_val)
# Evaluate SVR
mse_svr, rmse_svr, r2_svr = evaluate_model(svr_model, X_val, y_val)

# Print evaluation results
print("Linear Regression: MSE=", mse_lr, ", RMSE=", rmse_lr, ", R^2=", r2_lr)
print("Random Forest: MSE=", mse_rf, ", RMSE=", rmse_rf, ", R^2=", r2_rf)
print("SVR: MSE=", mse_svr, ", RMSE=", rmse_svr, ", R^2=", r2_svr)

# Model Selection based on validation performance
best_model = random_forest_model if r2_rf > max(r2_lr, r2_svr) else svr_model if r2_svr > r2_lr else linear_model

# Test the best model
mse_test, rmse_test, r2_test = evaluate_model(best_model, X_test, y_test)
print("Best Model Test Performance: MSE=", mse_test, ", RMSE=", rmse_test, ", R^2=", r2_test) 

# Save the preprocessed data to a CSV file
df.to_csv("preprocessed_city_day.csv", index=False)

# Generate and save accuracy charts
models = ['Linear Regression', 'Random Forest', 'SVR']
mse_values = [mse_lr, mse_rf, mse_svr]
rmse_values = [rmse_lr, rmse_rf, rmse_svr]
r2_scores = [r2_lr, r2_rf, r2_svr]

# Save Mean Squared Error (MSE) chart
plt.figure(figsize=(10, 6))
plt.bar(models, mse_values, color='skyblue')
plt.xlabel('Models')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('Model Comparison: Mean Squared Error (MSE)')
plt.savefig('mse_chart.png')
plt.close()

# Save Root Mean Squared Error (RMSE) chart
plt.figure(figsize=(10, 6))
plt.bar(models, rmse_values, color='lightgreen')
plt.xlabel('Models')
plt.ylabel('Root Mean Squared Error (RMSE)')
plt.title('Model Comparison: Root Mean Squared Error (RMSE)')
plt.savefig('rmse_chart.png')
plt.close()

# Save R² Score chart
plt.figure(figsize=(10, 6))
plt.bar(models, r2_scores, color='coral')
plt.xlabel('Models')
plt.ylabel('R² Score')
plt.title('Model Comparison: R² Score')
plt.savefig('r2_chart.png')
plt.close()

# Generate and save residual error charts
# Predictions on validation data
y_pred_lr = linear_model.predict(X_val)
y_pred_rf = random_forest_model.predict(X_val)
y_pred_svr = svr_model.predict(X_val)

# Residuals (Actual - Predicted)
residuals_lr = y_val - y_pred_lr
residuals_rf = y_val - y_pred_rf
residuals_svr = y_val - y_pred_svr

# Plot Residual Error for Linear Regression
plt.figure(figsize=(10, 6))
plt.scatter(y_pred_lr, residuals_lr, color='skyblue', alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted AQI (Linear Regression)')
plt.ylabel('Residuals')
plt.title('Residual Errors: Linear Regression')
plt.savefig('residuals_lr.png')
plt.close()

# Plot Residual Error for Random Forest
plt.figure(figsize=(10, 6))
plt.scatter(y_pred_rf, residuals_rf, color='lightgreen', alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted AQI (Random Forest)')
plt.ylabel('Residuals')
plt.title('Residual Errors: Random Forest')
plt.savefig('residuals_rf.png')
plt.close()

# Plot Residual Error for SVR
plt.figure(figsize=(10, 6))
plt.scatter(y_pred_svr, residuals_svr, color='coral', alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted AQI (SVR)')
plt.ylabel('Residuals')
plt.title('Residual Errors: SVR')
plt.savefig('residuals_svr.png')
plt.close()

# Generate and save time series analysis of residuals
# Convert index to a datetime type for time series plotting
df_val = X_val.copy()
df_val['Date'] = pd.to_datetime(df['Date'].iloc[y_val.index])

# Add residuals to the validation dataframe
df_val['Residuals_LR'] = residuals_lr
df_val['Residuals_RF'] = residuals_rf
df_val['Residuals_SVR'] = residuals_svr

# Sort by Date for time series analysis
df_val = df_val.sort_values(by='Date')

# Plot Time Series of Residuals for Linear Regression
plt.figure(figsize=(14, 6))
plt.plot(df_val['Date'], df_val['Residuals_LR'], color='skyblue', label='Residuals (Linear Regression)')
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Residuals')
plt.title('Time Series of Residuals: Linear Regression')
plt.legend()
plt.savefig('timeseries_residuals_lr.png')
plt.close()

# Plot Time Series of Residuals for Random Forest
plt.figure(figsize=(14, 6))
plt.plot(df_val['Date'], df_val['Residuals_RF'], color='lightgreen', label='Residuals (Random Forest)')
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Residuals')
plt.title('Time Series of Residuals: Random Forest')
plt.legend()
plt.savefig('timeseries_residuals_rf.png')
plt.close()

# Plot Time Series of Residuals for SVR
plt.figure(figsize=(14, 6))
plt.plot(df_val['Date'], df_val['Residuals_SVR'], color='coral', label='Residuals (SVR)')
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Residuals')
plt.title('Time Series of Residuals: SVR')
plt.legend()
plt.savefig('timeseries_residuals_svr.png')
plt.close()

# End of code
