"import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load the CSV file
file_path = '/content/drive/MyDrive/ColabNotebooks/lstm/FYPdata.csv'
data = pd.read_csv(file_path)

# Convert 'datetime' to datetime object for filtering
data['datetime'] = pd.to_datetime(data['datetime'])

# Preprocess the training data (2009-01-01 to 2017-01-01)
train_data = data[(data['datetime'] >= '2009-01-01 00:00:00') & (data['datetime'] < '2017-01-01 00:00:00')]
scaler_train = MinMaxScaler(feature_range=(0, 1))
scaled_train_data = scaler_train.fit_transform(train_data[['Temperature', 'ENERGY_Produced', 'hour', 'dayofweek', 'month']])
scaled_train_data_df = pd.DataFrame(scaled_train_data, columns=['Temperature', 'ENERGY_Produced', 'hour', 'dayofweek', 'month'])

# Advanced feature engineering: Creating lag features
for i in [1, 2, 3, 24, 48]:
    scaled_train_data_df[f'Temperature_lag{i}'] = scaled_train_data_df['Temperature'].shift(i)
    scaled_train_data_df[f'Energy_Produced_lag{i}'] = scaled_train_data_df['ENERGY_Produced'].shift(i)
scaled_train_data_df.dropna(inplace=True)

# Split training data into features (X) and target (y)
X_train = scaled_train_data_df.drop(columns=['ENERGY_Produced'])
y_train = scaled_train_data_df['ENERGY_Produced']

# Random Forest Model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Preprocess the testing data (2017-01-01 to 2018-12-31)
test_data = data[(data['datetime'] >= '2017-01-01 00:00:00') & (data['datetime'] <= '2018-12-31 23:00:00')]
scaled_test_data = scaler_train.transform(test_data[['Temperature', 'ENERGY_Produced', 'hour', 'dayofweek', 'month']])
scaled_test_data_df = pd.DataFrame(scaled_test_data, columns=['Temperature', 'ENERGY_Produced', 'hour', 'dayofweek', 'month'])

# Apply the same lag feature engineering to the test data
for i in [1, 2, 3, 24, 48]:
    scaled_test_data_df[f'Temperature_lag{i}'] = scaled_test_data_df['Temperature'].shift(i)
    scaled_test_data_df[f'Energy_Produced_lag{i}'] = scaled_test_data_df['ENERGY_Produced'].shift(i)
scaled_test_data_df.dropna(inplace=True)

# Prepare testing data for prediction
X_test = scaled_test_data_df.drop(columns=['ENERGY_Produced'])

# Make predictions
predictions = model.predict(X_test)

# Invert the normalization for the 'ENERGY_Produced' column
num_original_features = scaled_train_data.shape[1]
dummy_array = np.zeros((len(predictions), num_original_features))
dummy_array[:, 1] = predictions
y_pred_actual = scaler_train.inverse_transform(dummy_array)[:, 1]

# Output and Analysis
test_data_aligned = test_data.iloc[len(test_data) - len(predictions):].copy()
test_data_aligned['predicted_energy'] = y_pred_actual

new_csv_file_path = '/content/drive/MyDrive/ColabNotebooks/lstm/predicted_data_random_forest_model_scenario.csv'
test_data_aligned.to_csv(new_csv_file_path, index=False)

print(f""Scenario Predictions saved to {new_csv_file_path}"")"
