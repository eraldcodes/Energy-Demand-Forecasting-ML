"import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D
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

# Reshape input for Transformer
X_train_reshaped = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))

# Transformer Model
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = LayerNormalization(epsilon=1e-6)(res)
    x = Dense(ff_dim, activation=""relu"")(x)
    x = Dropout(dropout)(x)
    x = Dense(inputs.shape[-1])(x)
    return x + res

inputs = tf.keras.Input(shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2]))
x = transformer_encoder(inputs, head_size=64, num_heads=4, ff_dim=64, dropout=0.4)
x = GlobalAveragePooling1D(data_format='channels_first')(x)
x = Dropout(0.4)(x)
outputs = Dense(1)(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Add EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

# Train the model
model.fit(X_train_reshaped, y_train, validation_split=0.2, epochs=50, batch_size=32, verbose=1, callbacks=[early_stopping])

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

new_csv_file_path = '/content/drive/MyDrive/ColabNotebooks/lstm/predicted_data_transformer_model_scenario.csv'
test_data_aligned.to_csv(new_csv_file_path, index=False)

print(f""Scenario Predictions saved to {new_csv_file_path}"")"
