import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
import yfinance as yf
from keras.models import load_model
import streamlit as st

# Added for custom tab-icon and tab-title
from PIL import Image

im = Image.open("img/favicon.png")
st.set_page_config(
    page_title = "Price Prophet",
    page_icon = im,
)

# Basic visualizations
st.title('Price Prophet : Stock trend prediction using stacked LSTM')

start = '2010-01-01'
end = '2019-12-31'
user_input = st.text_input('Enter a stock ticker', 'AAPL')
df = yf.download(user_input, start, end)

st.subheader('Stock data from 2010-2019')
st.write(df.describe())

# Time charts and moving averages
st.subheader('Time chart of closing price')
fig = plt.figure(figsize = (12, 6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Time chart of closing price with 100 MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize = (12, 6))
plt.plot(df.Close)
plt.plot(ma100)
st.pyplot(fig)

st.subheader('Time chart of closing price with 100 MA and 200 MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize = (12, 6))
plt.plot(df.Close)
plt.plot(ma100)
plt.plot(ma200)
st.pyplot(fig)

# Train-test split and preprocess for model
data_train = pd.DataFrame(df['Close'][0: int(len(df) * 0.70)])
data_test = pd.DataFrame(df['Close'][int(len(df) * 0.70): int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0, 1))
data_train_arr = scaler.fit_transform(data_train)

# Loading our model
model = load_model('keras_stock_model.h5')

# Testing closing prices via our model
past_100_days = data_train.tail(100)
final_df = past_100_days._append(data_test, ignore_index = True)
data_input = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, data_input.shape[0]):
    x_test.append(data_input[i - 100: i])
    y_test.append(data_input[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

# Predicting and plotting
y_pred = model.predict(x_test)

scale_factor = scaler.scale_
scale_factor = 1 / scale_factor[0]
y_pred = y_pred * scale_factor
y_test = y_test * scale_factor

st.title('Original vs Predictions')
fig = plt.figure(figsize = (12, 6))
plt.plot(y_test, 'b', label = 'Original')
plt.plot(y_pred, 'r', label = 'Predicted')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig)
