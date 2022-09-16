import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_option_menu import option_menu
import yfinance as yf

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
        options=["Download Stock Data", "Visualisation", "Upload Your Data"],
        menu_icon=["meta"],
        icons=["cloud-download-fill", "graph-up-arrow", "cloud-upload-fill"],
        default_index=0,
    )

df = pd.read_csv(
    'GOOGL.csv',
    engine = 'python',

)

st.markdown('[![Typing SVG](https://readme-typing-svg.demolab.com?font=Fira+Code&pause=1000&color=FF4B4B&width=435&lines=Hello+Techies;Fetch+Stocks;Visualize;%40chinmay29hub)](https://git.io/typing-svg)')



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
    company = st.text_input('Company Name', placeholder="eg : GOOGL, APPL, etc", type="default", autocomplete=None)
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
            
            




