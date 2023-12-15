import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('data.csv')

# Convert timestamp to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Set timestamp as index
df.set_index('timestamp', inplace=True)

# Feature Engineering
df['hour'] = df.index.hour
df['day_of_week'] = df.index.dayofweek
df['month'] = df.index.month

# Handle Missing Data (Replace NaN values with linear interpolation)
df.interpolate(inplace=True)

# Outlier Detection (Replace outliers with interpolated values)
outliers = np.abs(df['value'] - df['value'].mean()) > 2 * df['value'].std()
df.loc[outliers, 'value'] = np.nan
df['value'].interpolate(inplace=True)

# Time Series Decomposition using seasonal_decompose
result = seasonal_decompose(df['value'], period=5)  # Adjust the period as needed
df['trend'] = result.trend
df['seasonal'] = result.seasonal
df['residual'] = result.resid

# Train-Test Split
n = 5  # Assuming you want to predict the last 5 periods
train = df.iloc[:-n]
test = df.iloc[-n:]

# Feature Selection
X_train, y_train = train.drop(['value', 'trend', 'seasonal', 'residual'], axis=1), train['value']
X_test, y_test = test.drop(['value', 'trend', 'seasonal', 'residual'], axis=1), test['value']

# Ensemble Modeling (Random Forest + Gradient Boosting)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)

# Combine predictions from multiple models
def ensemble_predict(models, X):
    predictions = np.zeros(len(X))
    for model in models:
        predictions += model.predict(X)
    return predictions / len(models)

# Train models
rf_model.fit(X_train, y_train)
gb_model.fit(X_train, y_train)

# Make Predictions with Ensemble Model
ensemble_predictions = ensemble_predict([rf_model, gb_model], X_test)

# Evaluate the Ensemble Model
mse_ensemble = mean_squared_error(y_test, ensemble_predictions)
print(f'Mean Squared Error (Ensemble): {mse_ensemble}')

# Visualize Results
plt.figure(figsize=(16, 8))

plt.subplot(2, 1, 1)
plt.plot(df.index, df['value'], label='Actual', marker='o')
plt.plot(test.index, ensemble_predict([rf_model, gb_model], X_test), label='Ensemble Predicted', marker='o')
plt.title('Ensemble Time Series Forecasting')
plt.xlabel('Timestamp')
plt.ylabel('Value')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(df.index, df['trend'], label='Trend', marker='o')
plt.plot(df.index, df['seasonal'], label='Seasonal', marker='o')
plt.plot(df.index, df['residual'], label='Residual', marker='o')
plt.title('Time Series Decomposition')
plt.xlabel('Timestamp')
plt.legend()

plt.tight_layout()
plt.show()