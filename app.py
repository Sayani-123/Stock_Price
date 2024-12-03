import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from keras.models import load_model
import streamlit as st
import yfinance as yf

model = load_model('D:\stock_price\Stock Predictions Model.keras')

st.header("Stock Market Predictor")
stock =st.text_input('Enter Stock Symbol','GOOG')
start = '2012-01-01'
end = '2024-11-30'

data =yf.download(stock,start,end)
st.subheader('Stock Data')
st.write(data)

data_train =pd.DataFrame(data.Close[0:int(len(data)*0.80)]) 
data_test =pd.DataFrame(data.Close[int(len(data)*0.80):len(data)])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

pas_100_days = data_train.tail(100)
data_test = pd. concat([pas_100_days,data_test],ignore_index=True)
data_test_scale=scaler.fit_transform(data_test)

st.subheader('Price VS MA50')
ma_50=data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(8,6))
plt.plot(ma_50,'r',label="Moving Average of 50 days")
plt.plot(data.Close,'g',label="Price")
plt.legend()
plt.show()
st.pyplot(fig1)

st.subheader('Price VS MA50 VS MA100')
ma_100=data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(8,6))
plt.plot (ma_50,'r',label="Moving Average of 50 days")
plt.plot(ma_100,'b',label="Moving Average of 100 days")
plt.plot(data.Close,'g',label="Price")
plt.legend()
plt.show()
st.pyplot(fig2)

x=[]
y=[]
for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i,0])

x,y =np.array(x),np.array(y)

pred=model.predict(x)

scale = 1/scaler.scale_
pred = pred * scale
y=y*scale

st.subheader('Original Price VS Predicted Price')

fig3 = plt.figure(figsize=(8,6))
plt.plot (pred,'r',label='Orignal Price')
plt.plot(y,'g',label='Predicted Price')
plt.xlabel('Time(Days)')
plt.ylabel('Price')
plt.legend()
plt.show()
st.pyplot(fig3)


future_days = 100  # Adjust this value as needed for a longer forecast

# Prepare the base input (last 100 days of scaled data)
last_100_days = data_test_scale[-100:]
future_predictions = []

# Predict iteratively
for _ in range(future_days):
    # Reshape for model input
    last_100_days_reshaped = last_100_days.reshape(1, last_100_days.shape[0], 1)
    
    # Predict the next value
    next_pred = model.predict(last_100_days_reshaped)
    
    # Append the prediction to the future_predictions list
    future_predictions.append(next_pred[0, 0])
    
    # Update the last_100_days with the new prediction
    last_100_days = np.append(last_100_days, next_pred[0, 0])
    last_100_days = last_100_days[-100:]  # Keep the last 100 days

# Rescale future predictions to original price scale
future_predictions = np.array(future_predictions) * scale

# Create a timeline for plotting
future_timeline = np.arange(len(y), len(y) + future_days)

st.subheader('Original Price VS Predicted Price VS Future Price')
fig4=plt.figure(figsize=(8, 6))
plt.plot(y, 'g', label="Original Price")  # Original prices
plt.plot(range(len(y)), pred, 'r', label="Predicted Price")  # Predicted on test data
plt.plot(future_timeline, future_predictions, 'b', label="Future Price")  # Future predictions
plt.xlabel('Time(Days)')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()
st.pyplot(fig4)