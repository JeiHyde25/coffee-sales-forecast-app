import matplotlib.pyplot as plt
# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
import kagglehub
from sklearn.compose import ColumnTransformer
import os
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import streamlit as st

st.title("Coffee Sales Forecast")
st.write("This app predicts coffee sales based on historical data.")

@st.cache_data
def load_data():
  # Download the latest version
  path = kagglehub.dataset_download("ihelon/coffee-sales")
  path1 = path + "/index_1.csv"
  path2 = path + "/index_2.csv"

  # Load the dataset
  df1 = pd.read_csv(path1)
  df2 = pd.read_csv(path2)
  df = pd.concat([df1, df2], ignore_index=True)

  return df

df = load_data()


def preprocess_data(df):
    # Convert datetime
    df['datetime'] = pd.to_datetime(df['datetime'], format='mixed', errors='coerce')
    
    # Group by date and sum money
    daily_sales = df.groupby('date')['money'].sum().reset_index()
    daily_sales['date'] = pd.to_datetime(daily_sales['date'])
    
    # Calculate days since start
    start_date = daily_sales['date'].min()
    daily_sales['days_since_start'] = (daily_sales['date'] - start_date).dt.days
    
    return daily_sales

daily_sales = preprocess_data(df)

# Split the dataset into an 80-20 training-test set
X = daily_sales[['days_since_start']]
y = daily_sales['money']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(daily_sales):
    X = daily_sales[['days_since_start']]
    y = daily_sales['money']
    
    model = LinearRegression()
    model.fit(X, y)
    return model

# Train the model
model = train_model(daily_sales)

# Predict
y_pred = model.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Save to txt file
os.makedirs("outputs", exist_ok=True)
with open('outputs/model_metrics.txt', 'w') as f:
    f.write(f"Mean Absolute Error: {mae}\n")
    f.write(f"Mean Squared Error: {mse}\n")
    f.write(f"R2 Score: {r2}")

# Plot
plt.scatter(X_test, y_test, color='green', label='Actual')
plt.scatter(X_test, y_pred, color='red', label='Predicted')


# In[6]:


last_day = daily_sales['days_since_start'].max()
future_days = pd.DataFrame({'days_since_start': range(last_day + 1, last_day + 8)})

future_predictions = model.predict(future_days)

future_days['predicted_sales'] = future_predictions
future_days['date'] = pd.to_datetime(daily_sales['date'].max()) + pd.to_timedelta(future_days['days_since_start'] - daily_sales['days_since_start'].max(), unit='d')

# Save predictions to CSV
future_days[['date', 'predicted_sales']].to_csv('output/future_predictions.csv', index=False)

# Visualize future broadcasts
plt.plot(future_days['date'], future_days['predicted_sales'], marker='o', color='blue')
plt.title('Prediction for the next 7 days')
plt.xlabel('Date')
plt.ylabel('Predicted Sales')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

st.header('Sales Prediction')

# Add a date input
prediction_date = st.date_input(
    "Select a date to predict sales:",
    min_value=daily_sales['date'].min(),
    value=daily_sales['date'].max() + pd.Timedelta(days=1)
)

if st.button('Predict Sales'):
    # Calculate days since start for the prediction date
    days_since_start = (prediction_date - daily_sales['date'].min().date()).days
    
    # Make prediction
    prediction = model.predict([[days_since_start]])[0]
    
    st.write(f"Predicted sales for {prediction_date}: ${prediction:.2f}")

st.header('Historical Sales Visualization')

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(daily_sales['date'], daily_sales['money'], alpha=0.5)
ax.set_xlabel('Date')
ax.set_ylabel('Sales ($)')
ax.set_title('Historical Coffee Sales')
plt.xticks(rotation=45)
st.pyplot(fig)

st.header('Future Sales Forecast')

n_days = st.slider('Number of days to forecast:', 1, 30, 7)

last_day = daily_sales['days_since_start'].max()
future_days = pd.DataFrame({'days_since_start': range(last_day + 1, last_day + n_days + 1)})

future_predictions = model.predict(future_days)
future_days['predicted_sales'] = future_predictions
future_days['date'] = pd.to_datetime(daily_sales['date'].max()) + pd.to_timedelta(
    future_days['days_since_start'] - daily_sales['days_since_start'].max(), unit='d'
)

fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.plot(future_days['date'], future_days['predicted_sales'], marker='o')
ax2.set_xlabel('Date')
ax2.set_ylabel('Predicted Sales ($)')
ax2.set_title(f'Sales Forecast for Next {n_days} Days')
plt.xticks(rotation=45)
st.pyplot(fig2)

# Show predictions in a table
st.write('Detailed Forecast:')
forecast_df = future_days[['date', 'predicted_sales']].copy()
forecast_df.columns = ['Date', 'Predicted Sales ($)']
st.dataframe(forecast_df)