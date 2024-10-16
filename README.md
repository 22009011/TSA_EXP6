## DEVELOPED BY: Thanjiyappan k
## REGISTER NO: 212222240108
## DATE:


# Ex.No: 6               HOLT WINTERS METHOD

### AIM:
To forecast sales using the Holt-Winters method and calculate the Test and Final Predictions.

### ALGORITHM:
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as
datetime, and perform some initial data exploration
3. You group the data by date and resample it to a monthly frequency (beginning of the month
4. You plot the time series data
5. You import the necessary 'statsmodels' libraries for time series analysis
6. You decompose the time series data into its additive components and plot them:
7. You calculate the root mean squared error (RMSE) to evaluate the model's performance
8. You calculate the mean and standard deviation of the entire sales dataset, then fit a Holt-
Winters model to the entire dataset and make future predictions
9. You plot the original sales data and the predictions
### PROGRAM:


```python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error

file_path = 'ev.csv'  # Path to the uploaded dataset
data = pd.read_csv(file_path)

yearly_data = data.groupby('year')['value'].sum()

plt.figure(figsize=(10, 5))
plt.plot(yearly_data.index, yearly_data.values, label='Yearly Data')
plt.title('Yearly Data: Value Over Time')
plt.xlabel('Year')
plt.ylabel('Value')
plt.legend()
plt.show()

train_size = int(len(yearly_data) * 0.8)
train, test = yearly_data[:train_size], yearly_data[train_size:]

model = ExponentialSmoothing(train, trend="add", seasonal=None, seasonal_periods=1)  # No seasonality in yearly data
fit = model.fit()

predictions = fit.forecast(len(test))

rmse = np.sqrt(mean_squared_error(test, predictions))
print(f'Test RMSE: {rmse}')

final_model = ExponentialSmoothing(yearly_data, trend="add", seasonal=None, seasonal_periods=1)  # No seasonality
final_fit = final_model.fit()

future_steps = 10
final_forecast = final_fit.forecast(steps=future_steps)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(yearly_data.index[:train_size], train, label='Training Data', color='blue')
plt.plot(yearly_data.index[train_size:], test, label='Test Data', color='green')
plt.plot(yearly_data.index[train_size:], predictions, label='Predictions', color='orange')
plt.title('Test Predictions')
plt.xlabel('Year')
plt.ylabel('Value')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(yearly_data.index, yearly_data.values, label='Original Data', color='blue')
plt.plot(range(yearly_data.index[-1] + 1, yearly_data.index[-1] + 1 + future_steps), 
         final_forecast, label='Final Forecast', color='orange')
plt.title('Final Predictions')
plt.xlabel('Year')
plt.ylabel('Value')
plt.legend()

plt.tight_layout()
plt.show()

```

### OUTPUT:

TEST AND FINAL PREDICTION:

![image](https://github.com/user-attachments/assets/4331151e-6734-466d-a467-988669fbbdc3)





### RESULT:
Thus, the program to forecast sales using the Holt-Winters method and calculate the Test and Final Prediction is executed successfully.
