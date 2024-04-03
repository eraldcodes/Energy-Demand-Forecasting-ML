"import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

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

# Reshape input for LSTM-CNN
X_train_reshaped = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))

# Hybrid LSTM-CNN Model
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(X_train_reshaped.shape[1], 1)))
model.add(MaxPooling1D(pool_size=2))
# Skip the Flatten layer here to maintain the 3D structure for LSTM
model.add(LSTM(50, return_sequences=False))
model.add(Flatten()) # Use Flatten after LSTM
model.add(Dense(50, activation='relu'))
model.add(Dense(1))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Add EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

# Split the training data
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train_reshaped, y_train, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train_split, y_train_split, validation_data=(X_val_split, y_val_split), epochs=50, batch_size=32, verbose=1, callbacks=[early_stopping])

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
X_test_reshaped = X_test.values.reshape((X_test.shape[0], X_test.shape[1], 1))

# Make predictions
predictions = model.predict(X_test_reshaped)

# Invert the normalization for the 'ENERGY_Produced' column
dummy_array = np.zeros((len(predictions), scaled_train_data.shape[1]))
dummy_array[:, 1] = predictions.reshape(-1)
y_pred_actual = scaler_train.inverse_transform(dummy_array)[:, 1]

# Output and Analysis
test_data_aligned = test_data.iloc[len(test_data) - len(predictions):].copy()
test_data_aligned['predicted_energy'] = y_pred_actual

new_csv_file_path = '/content/drive/MyDrive/ColabNotebooks/lstm/predicted_data_lstm_cnn_model_scenario.csv'
test_data_aligned.to_csv(new_csv_file_path, index=False)

print(f""Scenario Predictions saved to {new_csv_file_path}"")"
