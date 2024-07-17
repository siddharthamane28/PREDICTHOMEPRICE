import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer  # Importing SimpleImputer for handling missing values

# Step 1: Load and preprocess the data
data = pd.read_csv('Dataset.csv')

# Check for null values and replace them if necessary
if data.isnull().values.any():
    # Replace null values with mean (you can choose other strategies as well)
    imputer = SimpleImputer(strategy='mean')
    data[['area', 'bedrooms', 'parking']] = imputer.fit_transform(data[['area', 'bedrooms', 'parking']])

# Validate the data (X) and then predict (y)
X = data[['area', 'bedrooms', 'parking']]  # Features
y = data['price']  # Target variable

# Standardize the features (optional but recommended)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 2: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 3: Train the model (Multiple Linear Regression)
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Step 4: Evaluate the model
# Predictions on training set and test set
train_preds = linear_model.predict(X_train)
test_preds = linear_model.predict(X_test)

# Calculate Mean Squared Error (MSE) as a metric
train_mse = mean_squared_error(y_train, train_preds)
test_mse = mean_squared_error(y_test, test_preds)

# Example prediction for a new house
new_house = np.array([[2000, 3, 1]])
scaled_new_house = scaler.transform(new_house)
predicted_price = linear_model.predict(scaled_new_house)
print(f"Predicted price for the new house (Area: 2000 sq units, Bedrooms: 3): ${predicted_price[0]:.5f}")

# Step 5: Print the coefficients with five digits
coefficients = linear_model.coef_
print("Coefficients:")
for coef in coefficients:
    print(f"{coef:.5f}")

# Step 6: Visualization: Scatter plot of predicted vs actual prices
plt.figure(figsize=(10, 6))

# Sort the test predictions and actual values by area for better visualization
sorted_indices = X_test[:, 0].argsort()
X_test_sorted = X_test[sorted_indices]
y_test_sorted = y_test.iloc[sorted_indices]
test_preds_sorted = test_preds[sorted_indices]

# Plot actual prices
plt.scatter(X_test_sorted[:, 0], y_test_sorted, color='blue', label='Actual Prices')

# Plot predicted prices
plt.scatter(X_test_sorted[:, 0], test_preds_sorted, color='red', label='Predicted Prices')

plt.xlabel('Scaled Area')
plt.ylabel('Price')
plt.title('Actual vs Predicted House Prices')
plt.legend()
plt.show()
