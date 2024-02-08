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
        crypto_currency = str(first+'-'+second)
        data = yf.download(tickers=crypto_currency, start=start, end=end)
        coc = data['Close']
        change = coc[-1] - coc[-2]
        st.metric(label="Change", value=coc[-1], delta=change)
        st.dataframe(data)
        st.session_state['button'] = False
        st.checkbox('Reload')
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
        st.dataframe(data)
        st.session_state['button'] = False


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