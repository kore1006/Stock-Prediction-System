import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from datetime import datetime

# Load the dataset (replace 'TCS.csv' with your file name or path)
try:
    data = pd.read_csv('TCS.csv')
except FileNotFoundError:
    print("Error: CSV file 'TCS.csv' not found.")
    exit(1)

# Parse the Date column with the correct format
data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%Y')

# Extract features (Year, Month, Day, Open, High, Low, Close) and target variable (Trend)
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day

# Prepare features (X) and target variable (y) for training
X = data[['Year', 'Month', 'Day', 'Open', 'High', 'Low', 'Close']]
y = data['trend']  # Assuming 'Trend' is the target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize RandomForestClassifier with adjusted parameters
clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

# Train the classifier
clf.fit(X_train, y_train)

# Evaluate model performance on training data
train_predictions = clf.predict(X_train)
train_accuracy = accuracy_score(y_train, train_predictions)
print("Training Accuracy:", train_accuracy)

# Evaluate model performance on testing data
test_predictions = clf.predict(X_test)
test_accuracy = accuracy_score(y_test, test_predictions)
print("Testing Accuracy:", test_accuracy)

# Additional evaluation metrics
print("Testing Classification Report:")
print(classification_report(y_test, test_predictions))

print("Testing Confusion Matrix:")
print(confusion_matrix(y_test, test_predictions))

# Example: Predicting on user-provided date
date_input = input("Enter a date (format: MM/DD/YYYY): ")

try:
    user_date = datetime.strptime(date_input, "%m/%d/%Y")
except ValueError:
    print("Error: Invalid date format. Please use the format MM/DD/YYYY.")
    exit(1)

# Prepare input data for prediction using the user-provided date
input_data = {
    'Date': [user_date],
    'Open': [0.0],   # Example values for other features (not used for prediction)
    'High': [0.0],
    'Low': [0.0],
    'Close': [0.0]
}

input_df = pd.DataFrame(input_data)
input_df['Year'] = input_df['Date'].dt.year
input_df['Month'] = input_df['Date'].dt.month
input_df['Day'] = input_df['Date'].dt.day

# Make predictions on the input data
new_predictions = clf.predict(input_df[['Year', 'Month', 'Day', 'Open', 'High', 'Low', 'Close']])
print("Raw Predictions:", new_predictions)

# Display the predicted trend
predicted_trend = new_predictions[0]
if predicted_trend == 1:
    print("Predicted Trend for", date_input, ": Upward (Bullish)")
else:
    print("Predicted Trend for", date_input, ": Downward or Neutral (Bearish)")
