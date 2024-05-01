import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from tkinter import *
from tkinter import messagebox
from datetime import datetime
import yfinance as yf
from PIL import Image, ImageTk
import os
from tkcalendar import DateEntry  # Importing DateEntry widget from tkcalendar module
import tkinter.ttk as ttk

# Load your dataset (replace 'TCS.csv' with the path and filename of your CSV file)
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

# Map trend to numerical labels (0 for bearish, 1 for bullish)
label_encoder = LabelEncoder()
data['trend'] = label_encoder.fit_transform(data['trend'])

# Prepare features (X) and target variable (y) for training
X = data[['Year', 'Month', 'Day', 'Open', 'High', 'Low', 'Close']]
y = data['trend']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize XGBoost Classifier with adjusted parameters
clf = xgb.XGBClassifier(objective='binary:logistic', random_state=42)

# Optionally perform hyperparameter tuning using GridSearchCV
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.1, 0.01],
    'n_estimators': [100, 200]
}

grid_search = GridSearchCV(clf, param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train, y_train)

best_clf = grid_search.best_estimator_

# Train the classifier on the full training data with the best parameters
best_clf.fit(X_train, y_train)

# Function to predict trend for user-provided date
def predict_trend():
    date_input = cal.get_date().strftime("%m/%d/%Y")  # Convert datetime.date to string
    try:
        user_date = datetime.strptime(date_input, "%m/%d/%Y")
    except ValueError:
        messagebox.showerror("Error", "Invalid date format. Please use the format MM/DD/YYYY.")
        return

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
    new_predictions = best_clf.predict(input_df[['Year', 'Month', 'Day', 'Open', 'High', 'Low', 'Close']])

    # Display the predicted trend
    if new_predictions[0] == 1:
        messagebox.showinfo("Prediction Result", f"Predicted Trend for {date_input}: Upward (Bullish)")
    else:
        messagebox.showinfo("Prediction Result", f"Predicted Trend for {date_input}: Downward or Neutral (Bearish)")

# Function to get stock data from Yahoo Finance
def get_stock_data(stock_identifier):
    try:
        stock = yf.Ticker(f"{stock_identifier}.NS")
        data = stock.history(period='1d')
        
        if not data.empty:
            current_price = data['Close'][0]
            opening_price = data['Open'][0]
            highest_price = data['High'].max()
            lowest_price = data['Low'].min()
            
            # Predict trend for the current date
            date_input = datetime.today().strftime("%m/%d/%Y")
            user_date = datetime.strptime(date_input, "%m/%d/%Y")
            
            # Prepare input data for prediction using the current date
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
            prediction = best_clf.predict(input_df[['Year', 'Month', 'Day', 'Open', 'High', 'Low', 'Close']])
            predicted_price = current_price * (1 + 0.01 if prediction[0] == 1 else -0.01)  # Adjust the predicted price
            
            stock_data = {
                'Stock Identifier': stock_identifier,
                'Current Price (INR)': current_price,
                'Opening Price (INR)': opening_price,
                'Highest Price (INR)': highest_price,
                'Lowest Price (INR)': lowest_price,
                'Predicted Price (INR)': predicted_price  # Include the predicted price
            }
            
            return stock_data
        else:
            raise ValueError("No data available for the stock")
    
    except Exception as e:
        messagebox.showerror("Error", f"Failed to fetch data for {stock_identifier}: {e}")
        return None

# Function to track stock prices
def track_stock_prices():
    stock_identifiers = entry.get().strip().split(',')
    
    if not stock_identifiers:
        messagebox.showerror("Error", "Please enter at least one stock identifier")
        return
    
    stock_data_list = []

    for identifier in stock_identifiers:
        stock_data = get_stock_data(identifier)
        
        if stock_data is not None:
            stock_data_list.append(stock_data)
    
    if stock_data_list:
        save_to_excel(stock_data_list)
        messagebox.showinfo("Success", "Stock data tracked and saved successfully")

# Function to save stock data to Excel
def save_to_excel(stock_data_list):
    file_name = 'stock_prices_INR.xlsx'
    
    try:
        if os.path.exists(file_name):
            # Read existing data
            existing_df = pd.read_excel(file_name)
            df = pd.DataFrame(stock_data_list)
            # Append new data to existing DataFrame
            df_combined = pd.concat([existing_df, df], ignore_index=True)
            df_combined.to_excel(file_name, index=False)
        else:
            # Create new Excel file
            df = pd.DataFrame(stock_data_list)
            df.to_excel(file_name, index=False)
    
    except Exception as e:
        messagebox.showerror("Error", f"Failed to save data to Excel: {e}")

# Function to set background image
def set_background_image(window):
    try:
        image = Image.open("background.jpg")
        image = image.resize((500, 450))
        photo = ImageTk.PhotoImage(image)
        background_label = Label(window, image=photo)
        background_label.image = photo
        background_label.place(x=0, y=0, relwidth=1, relheight=1)
        
        label1.lift()
        entry.lift()
        cal.lift()
        # predict_button.lift()
        label2.lift()
        label3.lift()
        label4.lift()
        label5.lift()
        label6.lift()
        label7.lift()
        # track_button.lift()
        # show_data_button.lift()
        predict_track_show_button.lift()
        
    except Exception as e:
        messagebox.showerror("Error", f"Failed to set background image: {e}")

# Function to display fetched data in a tabular form
def show_data():
    stock_identifiers = entry.get().strip().split(',')
    if not stock_identifiers:
        messagebox.showerror("Error", "Please enter at least one stock identifier")
        return
    
    data_list = []
    columns = ["Stock Identifier", "Current Price (INR)", "Opening Price (INR)", "Highest Price (INR)", "Lowest Price (INR)", "Predicted Price (INR)", "Trend Prediction"]
    
    for identifier in stock_identifiers:
        stock_data = get_stock_data(identifier)
        if stock_data is not None:
            # Predict trend for the current date
            date_input = datetime.today().strftime("%m/%d/%Y")
            user_date = datetime.strptime(date_input, "%m/%d/%Y")
            
            # Prepare input data for prediction using the current date
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
            prediction = best_clf.predict(input_df[['Year', 'Month', 'Day', 'Open', 'High', 'Low', 'Close']])
            prediction_text = "Upward (Bullish)" if prediction[0] == 1 else "Downward or Neutral (Bearish)"
            
            data_list.append([
                stock_data['Stock Identifier'],
                stock_data['Current Price (INR)'],
                stock_data['Opening Price (INR)'],
                stock_data['Highest Price (INR)'],
                stock_data['Lowest Price (INR)'],
                stock_data['Predicted Price (INR)'],  # Include the predicted price
                prediction_text
            ])
    
    if data_list:
        # Create a new window for displaying data
        data_window = Toplevel()
        data_window.title("Fetched Stock Data")
        
        # Create Treeview widget for tabular display
        tree = ttk.Treeview(data_window, columns=columns, show='headings')
        for col in columns:
            tree.heading(col, text=col)
        
        # Insert fetched data into the Treeview
        for i, row in enumerate(data_list, start=1):
            tree.insert("", "end", values=row)
        
        tree.pack(expand=True, fill='both')

# Function to perform predict, track price, and show data actions
def predict_track_show():
    predict_trend()
    track_stock_prices()
    show_data()

# Create GUI window
window = Tk()
window.geometry("500x450")
window.title("Stock Price Tracker & Trend Predictor (INR)")

# GUI components
label1 = Label(window, text="Select a date:")
label1.grid(row=20, column=28, padx=15, pady=5)

# Using DateEntry widget for date selection
cal = DateEntry(window, background='darkblue', foreground='white', borderwidth=2)
cal.grid(row=22, column=28, padx=15, pady=5)

# Buttons and other components
# predict_button = Button(window, text="Predict Trend", command=predict_trend)
# predict_button.grid(row=24, column=28, padx=15, pady=5)

label2 = Label(window, text="Enter Stock Identifiers :")
label2.grid(row=30, column=28, padx=15, pady=5)

entry = Entry(window, width=50)
entry.grid(row=40, column=28, padx=15, pady=5)

# track_button = Button(window, text="Track Prices", command=track_stock_prices)
# track_button.grid(row=50, column=28, padx=15, pady=5)

# show_data_button = Button(window, text="Show Data", command=show_data)
# show_data_button.grid(row=60, column=28, padx=15, pady=5)

predict_track_show_button = Button(window, text="Show Data", command=predict_track_show)
predict_track_show_button.grid(row=70, column=28, padx=15, pady=5)

label3 = Label(window, text="Created by: ")
label3.grid(row=80, column=2, padx=15, pady=5)
label4 = Label(window, text="Vedant Hiwarde ")
label4.grid(row=82, column=2, padx=15, pady=5)
label5 = Label(window, text="Sahil Nikam ")
label5.grid(row=84, column=2, padx=15, pady=5)
label6 = Label(window, text="Prajwal Alekar ")
label6.grid(row=86, column=2, padx=15, pady=5)
label7 = Label(window, text="Prajwal Nimbalkar ")
label7.grid(row=88, column=2, padx=15, pady=5)

# Set background image
set_background_image(window)

# Start the main GUI event loop
window.mainloop()
