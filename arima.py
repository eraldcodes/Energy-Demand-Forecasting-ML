"import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt

# Load the CSV file
file_path = '/content/drive/MyDrive/ColabNotebooks/lstm/FYPdata.csv'
data = pd.read_csv(file_path)

# Convert 'datetime' to datetime object for filtering and set it as index
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

# Preprocess the training data (2009-01-01 to 2017-01-01)
train_data = data['2009-01-01':'2017-01-01']

# Assuming 'ENERGY_Produced' is the target variable
y_train = train_data['ENERGY_Produced']

# Fit an ARIMA model
# Note: You might need to experiment with different orders (p, d, q) for best results
model = ARIMA(y_train, order=(5,1,0))  # Example order, adjust based on your data
model_fit = model.fit()

# Preprocess the testing data (2017-01-01 to 2018-12-31)
test_data = data['2017-01-01':'2018-12-31']
y_test = test_data['ENERGY_Produced']

# Make predictions
predictions = model_fit.forecast(steps=len(test_data))  # Ensure the steps match the length of test_data

# Create a DataFrame for storing actual and predicted values
results = pd.DataFrame({'Actual': y_test, 'Predicted': predictions.values}, index=test_data.index)

# Calculate RMSE
rmse = sqrt(mean_squared_error(y_test, predictions))
print('Test RMSE: %.3f' % rmse)

# Save the results to a new CSV file
new_csv_file_path = '/content/drive/MyDrive/ColabNotebooks/lstm/predicted_data_arima_model_scenario.csv'
results.to_csv(new_csv_file_path)

print(f""Scenario Predictions saved to {new_csv_file_path}"")"
