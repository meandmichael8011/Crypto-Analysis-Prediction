#!/usr/bin/env python
# coding: utf-8

# In[1]:
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential
import yfinance as yf

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential
import yfinance as yf
import streamlit as st


# In[2]:


from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM


# In[3]:
option_names = ["Cryptocurrency Analysis", "Stock Analysis"]

#company = input("Insert a tick: ")
st.title('''Stock Analysis WebApp''')
st.subheader("Find the latest stock/currency info!")
st.image('stock.jpg')
st.text("Get the latest data for currency exchange and/or the latest Stock Data!")

st.markdown("You can find all the tickers " + "[here](https://finance.yahoo.com/currencies)" + ".")

option = st.radio("Select a type", option_names)

if option == "Cryptocurrency Analysis":
    start = dt.datetime(2012,1,1)
    end = dt.datetime.now()
    first = st.text_input("Pass in the first currency in a pair: ")
    second = st.text_input("Pass in the second currency in a pair: ")
    if st.button("Analyze"):
        try:
            crypto_currency = str(first+'-'+second)
            data = yf.download(tickers=crypto_currency, start=start, end=end)
            coc = data['Close']
            change = coc[-1] - coc[-2]
        except IndexError:
            first1=first
            second1=second
            first=second1
            second=first1
            crypto_currency = str(first + '-' + second)
            data = yf.download(tickers=crypto_currency, start=start, end=end)
            coc = data['Close']
            change = coc[-1] - coc[-2]
        st.metric(label="Change", value=coc[-1], delta=change)
        st.dataframe(data.iloc[::-1])
        st.session_state['button'] = False
        st.write("Your Next Day prediction is being prepared. Please wait...")
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
        prediction_days = 60
        x_train = []
        y_train = []
        for x in range(prediction_days, len(scaled_data)):
            x_train.append(scaled_data[x - prediction_days:x, 0])
            y_train.append(scaled_data[x, 0])
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(x_train, y_train, epochs=45, batch_size=32)
        test_start = dt.datetime(2023, 1, 1)
        test_end = dt.datetime.now()
        test_data = yf.download(tickers=crypto_currency, start=test_start, end=test_end)
        actual_prices = test_data['Close'].values
        total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)
        model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
        model_inputs = model_inputs.reshape(-1, 1)
        model_inputs = scaler.transform(model_inputs)
        x_test = []
        for x in range(prediction_days, len(model_inputs)):
            x_test.append(model_inputs[x - prediction_days:x, 0])
        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        predicted_prices = model.predict(x_test)
        predicted_prices = scaler.inverse_transform(predicted_prices)
        real_data = [model_inputs[len(model_inputs) + 1 - prediction_days:len(model_inputs) + 1, 0]]
        real_data = np.array(real_data)
        real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))
        prediction = model.predict(real_data)
        prediction = scaler.inverse_transform(prediction)
        st.metric(label="Next Day Prediction", value=float(prediction), delta=float(prediction - coc[-1]))
        print(f"Prediction: {prediction}")


if option == "Stock Analysis":
    start = dt.datetime(2012, 1, 1)
    end = dt.datetime.now()
    first = st.text_input("Pass in the ticker: ")
    crypto_currency = str(first)
    if st.button("Analyze"):
        data = yf.download(tickers=crypto_currency, start=start, end=end)
        coc = data['Close']
        change = coc[-1] - coc[-2]
        st.metric(label="Change", value=coc[-1], delta=change)
        st.dataframe(data.style.set_properties(**{'background-color': 'white',
                           'color': 'green'}))
        st.session_state['button'] = False
        st.write("Your Next Day prediction is being prepared. Please wait...")
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
        prediction_days = 60
        x_train = []
        y_train = []
        for x in range(prediction_days, len(scaled_data)):
            x_train.append(scaled_data[x - prediction_days:x, 0])
            y_train.append(scaled_data[x, 0])
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(x_train, y_train, epochs=45, batch_size=32)
        test_start = dt.datetime(2023, 1, 1)
        test_end = dt.datetime.now()
        test_data = yf.download(tickers=crypto_currency, start=test_start, end=test_end)
        actual_prices = test_data['Close'].values
        total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)
        model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
        model_inputs = model_inputs.reshape(-1, 1)
        model_inputs = scaler.transform(model_inputs)
        x_test = []
        for x in range(prediction_days, len(model_inputs)):
            x_test.append(model_inputs[x - prediction_days:x, 0])
        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        predicted_prices = model.predict(x_test)
        predicted_prices = scaler.inverse_transform(predicted_prices)
        real_data = [model_inputs[len(model_inputs) + 1 - prediction_days:len(model_inputs) + 1, 0]]
        real_data = np.array(real_data)
        real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))
        prediction = model.predict(real_data)
        prediction = scaler.inverse_transform(prediction)
        st.metric(label="Next Day Prediction", value=float(prediction), delta=float(prediction - coc[-1]))
        print(f"Prediction: {prediction}")


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style/style.css")

animation_symbol="❄️"

st.markdown(
    f"""
    <div class="snowflake">{animation_symbol}</div>
    <div class="snowflake">{animation_symbol}</div>
    <div class="snowflake">{animation_symbol}</div>
    <div class="snowflake">{animation_symbol}</div>
    <div class="snowflake">{animation_symbol}</div>
    <div class="snowflake">{animation_symbol}</div>
    <div class="snowflake">{animation_symbol}</div>
    <div class="snowflake">{animation_symbol}</div>
    <div class="snowflake">{animation_symbol}</div>
    <div class="snowflake">{animation_symbol}</div>
    <div class="snowflake">{animation_symbol}</div>
    <div class="snowflake">{animation_symbol}</div>
    """,
    unsafe_allow_html=True,
)