import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_option_menu import option_menu
import yfinance as yf

import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from PIL import Image

from sklearn.preprocessing import MinMaxScaler
import numpy as np

st.set_page_config(
    page_title="Stockulize",
    page_icon=":bar_chart:",
    layout="wide"
)

hide_st_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

with st.sidebar:
    selected_y = option_menu(
        menu_title="Main Menu",
        options=["Download Stock Data", "Visualisation", "Upload Your Data", "Supported Companies", "ANN", "RNN", "RNN_2"],
        menu_icon=["meta"],
        icons=["cloud-download-fill", "graph-up-arrow", "cloud-upload-fill", "building"],
        default_index=0,
    )

df = pd.read_csv(
    'GOOGL.csv',
    engine = 'python',

)

st.markdown('![Typing SVG](https://readme-typing-svg.demolab.com?font=Fira+Code&size=25&pause=1000&color=FF4B4B&width=200&lines=Hello+Techies;Fetch+Stocks;Visualize;%40chinmay29hub)')



if selected_y == "Visualisation":
    st.subheader("Demo Graphs on Google Data")
    selected = option_menu(
        menu_title=None,
        options=["Open|Close", "Low|High", "Adj Close|Volume"],
        icons=["file-bar-graph", "file-bar-graph", "file-bar-graph"],
        default_index=0,
        orientation="horizontal",
    )

    fig_1 = px.line(df, x = 'Date', y = 'Open', title='Open')
    fig_2 = px.line(df, x = 'Date', y = 'Close', title='Close')
    fig_3 = px.line(df, x = 'Date', y = 'Low', title='Low')
    fig_4 = px.line(df, x = 'Date', y = 'High', title='High')
    fig_5 = px.line(df, x = 'Date', y = 'Adj Close', title='Adj Close')
    fig_6 = px.line(df, x = 'Date', y = 'Volume', title='Volume')

    left_column, right_column = st.columns(2)

    if selected == "Open|Close":
        left_column.plotly_chart(fig_1, use_container_width=True)
        right_column.plotly_chart(fig_2, use_container_width=True)

    if selected == "Low|High":
        left_column.plotly_chart(fig_3, use_container_width=True)
        right_column.plotly_chart(fig_4, use_container_width=True)

    if selected == "Adj Close|Volume":
        left_column.plotly_chart(fig_5, use_container_width=True)
        right_column.plotly_chart(fig_6, use_container_width=True)

if selected_y == "Download Stock Data":
    company = st.text_input('Company Name', placeholder="eg : GOOGL, AAPL, etc", type="default", autocomplete=None)
    start_date = st.date_input('Start Date', value=None, min_value=None, max_value=None, key=None)

    end_date = st.date_input('End Date', value=None, min_value=None, max_value=None, key=None)

    if st.button("Fetch"):
        data_df = yf.download(company, start_date, end_date)
        name = company + '.csv'
        d = data_df.to_csv(name)

        with open(name, "rb") as file:

            btn = st.download_button(

                label="Download Dataset",

                data=file,

                file_name=name,

                mime="text/csv",

                )

if selected_y == "Upload Your Data":
    st.subheader("Let's Visualize Your Data")
    data_file = st.file_uploader("Upload stock data as CSV",type=['csv'])
    if st.button("Process"):
        if data_file is not None:
            file_details = {"Filename":data_file.name,"FileType":data_file.type,"FileSize":data_file.size}
            st.write(file_details)

            df1 = pd.read_csv(data_file)
            st.dataframe(df1)

            fig_u_1 = px.line(df1, x = 'Date', y = 'Open', title='Open')
            fig_u_2 = px.line(df1, x = 'Date', y = 'Close', title='Close')
            fig_u_3 = px.line(df1, x = 'Date', y = 'Low', title='Low')
            fig_u_4 = px.line(df1, x = 'Date', y = 'High', title='High')
            fig_u_5 = px.line(df1, x = 'Date', y = 'Adj Close', title='Adj Close')
            fig_u_6 = px.line(df1, x = 'Date', y = 'Volume', title='Volume')
            
            left_column_2, right_column_2 = st.columns(2)
        
            
            left_column_2.plotly_chart(fig_u_1, use_container_width=True)
            right_column_2.plotly_chart(fig_u_2, use_container_width=True)

        
            left_column_2.plotly_chart(fig_u_3, use_container_width=True)
            right_column_2.plotly_chart(fig_u_4, use_container_width=True)

        
            left_column_2.plotly_chart(fig_u_5, use_container_width=True)
            right_column_2.plotly_chart(fig_u_6, use_container_width=True)
            
if selected_y == "Supported Companies":
    comp = pd.read_csv('support.csv')
    st.header("Supported Companies")
    st.subheader("You can refer to the below data frame for getting the company sysmbol.")

    col_list = list(comp["Name"])
    col_list = tuple(col_list)

    option = st.selectbox(
    'Choose the company', col_list)


    st.dataframe(comp)

if selected_y == "ANN":
    dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
    X_test = dataset_test.iloc[:-1, 1:2].values
    y_test = dataset_test.iloc[1:, 1:2].values
    ann = tf.keras.models.load_model('ann.h5')
    y_pred = ann.predict(X_test)
    fig = Figure()
    plt = fig.add_subplot(1, 1, 1)
    plt.plot(y_test, color='red', label='Real Google Stock Price')
    plt.plot(y_pred, color='blue', label='Predicted Google Stock Price')
    plt.set_title('Google Stock Price Prediction')
    plt.set_xlabel('Time')
    plt.set_ylabel('Google Stock Price')
    fig.savefig('img/annGraph.png')
    
    image = Image.open('img/annGraph.png')

    st.image(image, caption='ANN')

if selected_y == "RNN":
    regressor = tf.keras.models.load_model('rnn_old.h5')
    dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
    # creating a numarray that contains the open price of the stock
    trainig_set = dataset_train.iloc[:, 1:2].values

    """### Feature Scaling"""

    # all stock prices will be between 0 and 1
    sc = MinMaxScaler(feature_range=(0, 1))
    training_set_scaled = sc.fit_transform(trainig_set)
    dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
    dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
    real_stock_price = dataset_test.iloc[:, 1:2].values

    """### Getting the predicted stock price of 2017"""

    dataset_total = pd.concat(
        (dataset_train['Open'], dataset_test['Open']), axis=0)
    inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
    inputs = inputs.reshape(-1, 1)

    inputs = sc.transform(inputs)
    X_test = []
    for i in range(60, 80):
        X_test.append(inputs[i-60:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    predicted_stock_price = regressor.predict(X_test)
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)
    # plotting the figure
    fig_rnn = Figure()
    plt = fig_rnn.add_subplot(1, 1, 1)
    plt.plot(real_stock_price, color='red', label='Real Google Stock Price')
    plt.plot(predicted_stock_price, color='blue',
             label='Predicted Google Stock Price')
    plt.set_title('Google Stock Price Prediction')
    plt.set_xlabel('Time')
    plt.set_ylabel('Google Stock Price')
    fig_rnn.savefig('img/RNN1Graph.png')

    image = Image.open('img/RNN1Graph.png')

    st.image(image, caption='RNN')

if selected_y == "RNN_2":
    regressor = tf.keras.models.load_model('rnn_new.h5')
    dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
    # creating a numarray that contains the open price of the stock
    my_train = dataset_train.iloc[:, 1:]
    training_set = my_train.replace(",", "", regex=True)

    from sklearn.preprocessing import MinMaxScaler
    # all stock prices will be between 0 and 1
    sc = MinMaxScaler(feature_range=(0, 1))
    training_set_scaled = sc.fit_transform(training_set)
    # print(training_set_scaled)

    X_train = []
    y_train = []
    # we will select the first 60 values to predict the first value and so on
    for i in range(60, 1258):
        # will put first 60 value in x
        X_train.append(training_set_scaled[i-60:i])
        # will put the value we will predict
        y_train.append(training_set_scaled[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
    real_stock_price = dataset_test.iloc[:, 1:2].values
    data_set1 = dataset_train.iloc[:, 1:]
    data_set2 = dataset_test.iloc[:, 1:]
    dataset_total = pd.concat((data_set1, data_set2), axis=0)
    dataset_total = dataset_total.replace(",", "", regex=True)
    inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
    inputs = sc.transform(inputs)
    X_test = []
    for i in range(60, 80):
        X_test.append(inputs[i-60:i])
    X_test = np.array(X_test)
    # # X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    predicted_stock_price = regressor.predict(X_test)
    # predicted_stock_price = sc.inverse_transform(predicted_stock_price)
    predicted_stock_price = predicted_stock_price*(1/1.86025746e-03)
    predicted_stock_price = predicted_stock_price+280
    # plotting the figure
    fig_rnnNew = Figure()
    plt = fig_rnnNew.add_subplot(1, 1, 1)
    plt.plot(real_stock_price, color='red', label='Real Google Stock Price')
    plt.plot(predicted_stock_price, color='blue',
            label='Predicted Google Stock Price')
    plt.set_title('Google Stock Price Prediction')
    plt.set_xlabel('Time')
    plt.set_ylabel('Google Stock Price')
    fig_rnnNew.savefig('img/RNN2Graph.png')

    image = Image.open('img/RNN2Graph.png')

    st.image(image, caption='RNN 2')







