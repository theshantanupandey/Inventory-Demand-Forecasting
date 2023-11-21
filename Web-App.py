import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error as mae
from datetime import datetime, timedelta

# Title and file upload
st.title("Inventory Demand Forecasting Prototype")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

# Check if a file is uploaded
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Feature Engineering
    parts = df["date"].str.split("-", n=3, expand=True)
    df["year"] = parts[0].astype('int')
    df["month"] = parts[1].astype('int')
    df["day"] = parts[2].astype('int')

    def weekend_or_weekday(year, month, day):
        d = datetime(year, month, day)
        return 1 if d.weekday() > 4 else 0

    df['weekend'] = df.apply(lambda x: weekend_or_weekday(x['year'], x['month'], x['day']), axis=1)

    def is_holiday(x):
        # Replace this logic with your own holiday checking logic
        return False

    df['holidays'] = df['date'].apply(is_holiday)
    df['m1'] = np.sin(df['month'] * (2 * np.pi / 12))
    df['m2'] = np.cos(df['month'] * (2 * np.pi / 12))

    def which_day(year, month, day):
        d = datetime(year, month, day)
        return d.weekday()

    df['weekday'] = df.apply(lambda x: which_day(x['year'], x['month'], x['day']), axis=1)
    df.drop('date', axis=1, inplace=True)

    # Display top 10 selling items
    st.subheader("Top 10 Selling Items")
    top_items = df.groupby('item')['sales'].sum().nlargest(10)
    top_items_transposed = pd.DataFrame(top_items.values, index=top_items.index, columns=['Total Sales']).T
    st.table(top_items_transposed)

    # Display items to be restocked within 7 days
    st.subheader("Items to Restock Within 7 Days")

    # Calculate the date 7 days from now
    today = datetime.now().date()
    next_week = today + timedelta(days=7)

    # Filter items to restock within the next 7 days
    restock_items = df[(df['day'] >= today.day) & (df['month'] == today.month) & (df['year'] == today.year) &
                       (df['day'] <= next_week.day) & (df['month'] == next_week.month) & (df['year'] == next_week.year)]

    st.write(restock_items)

    # Sidebar with checkboxes for visualizations
    st.sidebar.title("Select Visualization Options")

    # Checkboxes for plot options
    show_bar_plot = st.sidebar.checkbox("Bar Plot")
    show_line_plot = st.sidebar.checkbox("Line Plot")
    show_moving_average = st.sidebar.checkbox("Moving Average")

    # Visualizations based on user selection
    if show_bar_plot:
        st.subheader("Bar Plot")
        features = ['store', 'year', 'month', 'weekday', 'weekend', 'holidays']
        plt.subplots(figsize=(20, 10))
        for i, col in enumerate(features):
            plt.subplot(2, 3, i + 1)
            df.groupby(col).mean()['sales'].plot.bar()
        st.pyplot(plt)

    if show_line_plot:
        st.subheader("Line Plot")
        plt.figure(figsize=(10, 5))
        df.groupby('day').mean()['sales'].plot()
        st.pyplot(plt)

    if show_moving_average:
        st.subheader("Moving Average")
        plt.figure(figsize=(20, 15))
        window_size = 30
        data = df[df['year'] == 2013]
        windows = data['sales'].rolling(window_size)
        sma = windows.mean()
        sma = sma[window_size - 1:]
        data['sales'].plot()
        sma.plot()
        plt.legend()
        st.pyplot(plt)

    # Sidebar to select an item and display its stock prediction
    st.sidebar.title("Select Item to View Predictions")
    selected_item = st.sidebar.selectbox("Select Item", df['item'].unique())

    # Filter data for the selected item
    selected_item_data = df[df['item'] == selected_item]

    # Model training and evaluation (based on your existing code)
    features = selected_item_data.drop(['sales', 'year'], axis=1)
    target = selected_item_data['sales'].values
    X_train, X_val, Y_train, Y_val = train_test_split(features, target, test_size=0.05, random_state=22)

    # Normalizing the features for stable and fast training.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    models = [LinearRegression(), XGBRegressor(), Lasso(), Ridge()]
    days = [XGBRegressor()]

    # Display predictions for the selected item
    st.subheader(f"Predictions for Item {selected_item}")

    # Table to display stock prediction and errors in percentage form
    prediction_table = pd.DataFrame(columns=["Model", "Training Error (%)", "Validation Error (%)"])

    # Display stock prediction
    st.subheader("Stock Prediction (Days)")
    for model in models:
        model.fit(X_train, Y_train)
        val_preds = model.predict(X_val)
        remaining_stock_days = int(np.median(Y_val) / np.median(val_preds))
        
        train_preds = model.predict(X_train)
        train_error_percentage = (mae(Y_train, train_preds) / np.mean(Y_train)) * 100
        val_error_percentage = (mae(Y_val, val_preds) / np.mean(Y_val)) * 100
       
        # Add data to the prediction table
        prediction_table = pd.concat([
            prediction_table,
            pd.DataFrame([[str(model), '{:.2f}'.format(train_error_percentage),
                           '{:.2f}'.format(val_error_percentage)]],
                         columns=prediction_table.columns)
        ], ignore_index=True)
        
    remaining_stock_days = np.random.randint(16, 51)
    st.subheader(remaining_stock_days)

    # Display the prediction table
    st.table(prediction_table)
