import streamlit as st
from datetime import date

from utils.data_loader import download_stock_data
from utils.charts import line_chart
from utils.preprocess import DataPreprocessor
from utils.config import *
from train import train_model

from tensorflow.keras.models import load_model
import numpy as np

st.set_page_config(
    page_title="📈 Stock Time Series Analysis",
    layout='wide'
)

st.title("📈 Stock Price Forecasting Using LSTM")

st.sidebar.header("Configuration")

company = {"AAPL":"Alphabet Inc. (Apple)","MSFT":"Microsoft","GOOGL":"Google","TSLA":"Tesla","AMZN":"Amazon","NVDA":"Nvidia","RELIANCE.NS":"Reliance","TCS.NS":"TCS (Tata Consultancy Service)","INFY.NS":"Infosys","HDFCBANK.NS":"HDFC Bank"}
symbol = st.sidebar.selectbox("Stock Symbol" , company.keys())
start_date = st.sidebar.date_input("Start Date",value=date(2018,1,1))
end_date = st.sidebar.date_input("End Date",value=date.today())

if st.sidebar.button("Download Data"):
    with st.spinner("Downloading Data..."):
        df = download_stock_data(symbol,start_date,end_date)
    
    if df.empty:
        st.error("No Data Found!")
    else:
        st.session_state["df"] = df
        st.success(company.get(symbol)+" Download Data Complete")
        st.subheader(company.get(symbol)+" Dataset")
        st.dataframe(df , height=250)
        st.subheader(company.get(symbol)+" Data Statistics")
        st.write(df.describe())
        st.subheader(company.get(symbol)+" Closing Price")
        st.plotly_chart(
            line_chart(df , "Close"),
            use_container_width=True
        )

if st.button("Prepare Dataset"):
    if "df" in st.session_state:
        df = st.session_state["df"]
        preprocessor = DataPreprocessor(TIME_STEP)
        X,y = preprocessor.preprocess(df)
        preprocessor.save_scaler(SCALER_PATH)
        st.success(company.get(symbol)+" Dataset Prepared Successfully!")
        st.write("X Shape :",X.shape)
        st.write("y Shape :",y.shape)
        with st.spinner("Model training..."):
            train_model(symbol,start_date,end_date,X,y)
        st.success("Model Training Complete and Saved Successfully!")
    else:
        st.warning("Please Download the Stock Data First!")    


# Predict Next Closing Price
if st.button("Predict Next Closing Price"):

    if "df" in st.session_state:

        df = st.session_state["df"]

        preprocessor = DataPreprocessor(TIME_STEP)
        preprocessor.load_scaler(SCALER_PATH)

        model = load_model(MODEL_PATH)

        data = df[['Close']].tail(TIME_STEP)

        scaled_data = preprocessor.scaler.transform(data)

        X_test = np.array([scaled_data[:,0]])

        X_test = X_test.reshape(
            (X_test.shape[0], X_test.shape[1], 1)
        )

        prediction = model.predict(X_test)

        prediction = preprocessor.inverse_transform(prediction)

        st.subheader("Prediction Result")

        st.metric(
            "Predicted Next Closing Price",
            f"{prediction[0][0]:.2f}"
        )

    else:
        st.warning("Please Download the Stock Data First!")

