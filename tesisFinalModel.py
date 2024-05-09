import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib



# Load your dataset
data = pd.read_csv('tesisDatasetFinal.csv')

# Split the single column into multiple columns based on the semicolon delimiter
data_split = data[data.columns[0]].str.split(';', expand=True)

# Rename the columns for clarity
data_split.columns = ['paperAmount', 'recycledTree', 'newspaper', 'cardboard', 'mixed', 'inked', 'woody']

# Convert the columns to numeric type
data_split = data_split.apply(pd.to_numeric)

# Now you can use the splitted data for further processing
print(data_split.head())

X = data_split.drop(columns=['recycledTree'])
y = data_split['recycledTree']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

import matplotlib.pyplot as plt

# Plot the actual vs predicted values
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted')
plt.show()