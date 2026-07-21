import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import streamlit as st


# page config

st.set_page_config(
    page_title="Air Passengers Forecast",
    layout="wide"
)


# PATH
def load_model():
    return joblib.load('model/lr_model.pkl')

def load_results():
    df = pd.read_csv('model/results.csv')
    df['Month'] =pd.to_datetime(df['Month'])
    return df

def load_metrics():
    metrics ={}
    with open('model/metrics.txt','r') as f:
        for line in f:
            key, value = line.strip().split('=')
            metrics[key] = float(value)
    return metrics

def load_raw_data():
    df = pd.read_csv('dataset/AirPassengers.csv', header=0, names=['Month', 'Passengers'])
    df['Month'] = pd.to_datetime(df['Month'])
    df.set_index('Month', inplace=True)
    return df


model_exists = os.path.exists('model/lr_model.pkl')

if not model_exists:
    # Show a friendly error message if train.py hasn't been run yet
    st.error("Model not found! Please run train.py first.")
    st.code("python train.py", language="bash")
    st.stop()   # Stop the app here, don't show anything else


model   = load_model()
results = load_results()
metrics = load_metrics()
raw_df  = load_raw_data()

# Page title
st.title("Air Passengers — Time Series Forecast")
st.markdown("**Model:** Linear Regression with Lag Features | **Dataset:** Monthly Passengers (1949–1960)")
st.markdown("---")

# Slidebar 
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", [
    "Home",
    "Full Time Series",
    "Forecast Results",
    "Data Table"
])

#  Page 1 : Home
if page == "Home":

    col1,col2,col3,col4 = st.columns(4)

    col1.metric(
        label="MAE",
        value=f"{metrics['MAE']:.2f}",
        help="Mean Absolute Error — average prediction error in passengers"
    )

    col2.metric(
        label='RMSE',
        value=f"{metrics['RMSE']:2f}",
        help = "Root Mean Squared Error"
    )

    col3.metric(
        label="Total Data Points",
        value = len(raw_df)
    )

    col4.metric(
        label="Test Set Size",
        value= len(results)
    )

    st.markdown("-----")

# Page 2 
elif page == "Full Time Series":
    st.subheader("Full Air Passengers Time Series (1949-1960)")

    # Rolling Window Slider
    window = st.slider("Rolling Average Window (months)", min_value=3, max_value=24, value=12)
    roll_mean = raw_df['Passengers'].rolling(window=window).mean()

    fig, ax = plt.subplots(figsize=(12,3))

    ax.plot(raw_df.index, raw_df['Passengers'],
            label ='Actual Passengers',
            color = 'steelblue',
            linewidth=2
            )
    
    ax.plot(raw_df.index, roll_mean,
            label =f'{window}-Month Rolling Average',
            color ='orange',
            linewidth=2.5,
            linestyle='--'
            )
    ax.fill_between(raw_df.index, raw_df['Passengers'], alpha=0.08, color='steelblue')

    ax.set_title('Monthly Air Passengers with Rolling Average', fontsize=14, fontweight='bold')
    ax.set_xlabel('Year')
    ax.set_ylabel('Number of Passengers (thousands)')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Quick Stats
    st.markdown("---")
    st.subheader("Quick Stats")
    col1, col2, col3 = st.columns(3)
    col1.metric("Min Passengers", int(raw_df['Passengers'].min()))
    col2.metric("Max Passengers", int(raw_df['Passengers'].max()))
    col1.metric("Average", f"{raw_df['Passengers'].mean():.2f}")


# Page 3: Forecast Results

elif page == "Forecast Results":
    st.subheader("Model Forecast: Actual vs Predicted")

    # plot
    fig, ax = plt.subplots(figsize=(12,3))

    ax.plot(results['Month'],results['Actual'],
            label = 'Actual',
            color = 'steelblue',
            linewidth = 2,
            marker = 'o',
            markersize = 4)
    
    ax.plot(results['Month'], results['Predicted'],
            label = f"Predicted (MAE = {metrics['MAE']:.1f})",
            color ='orange',
            linewidth=2, 
            linestyle='--',
            marker='x',
            markersize=4
            )
    ax.set_title('Linear Regression - Actual vs Predicted', fontsize=14, fontweight='bold')
    ax.set_label('Month')
    ax.set_ylabel('Passengers (thousands)')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    st.pyplot(fig)
    plt.close()    
    
    # Error Distribution
    st.markdown("---")
    st.subheader("Prediction Error Distribrution")
    st.markdown("Error = Actual - Predicted. Closer to 0 = better predictions")

    fig2, ax2 = plt.subplots(figsize=(10,4))
    ax2.bar(results['Month'], results['Error'],
            color= results['Error'].apply(lambda x: 'steelblue' if x >= 0 else  'tomato'),
             alpha= 0.8)
    ax2.axhline(y=0, color='black', linewidth=1.2, linestyle='-')
    ax2.set_title('Prediction Error per Month  (Blue = over-predicted, Red = under-predicted)',
                  fontsize=12, fontweight='bold')
    ax2.set_xlabel('Month')
    ax2.set_ylabel('Error')
    ax2.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    # Metrics Box
    st.markdown("---")
    col1, col2 = st.columns(2)
    col1.info(f"**MAE = {metrics['MAE']:.2f}**\nOn average, prediction is off by {metrics['MAE']:.0f} passengers.")
    col2.info(f"**RMSE = {metrics['RMSE']:.2f}**\nPunishes big errors more than MAE.")
    
# Page 4: Data Table
elif page == "Data Table":
    st.subheader("Actual vs Predicted - Full Table")

    # Add a color hint column 
    df_display = results.copy()
    df_display['Month'] = df_display['Month'].dt.strftime('%Y-%m')

    st.dataframe(df_display,hide_index=True)

    st.markdown('--')
    col1,col2 = st.columns(2)
    col1.metric("Max Error (over-predicted)",f"{results['Error'].max():.2f}")
    col1.metric("Min Error (over-predicted)",f"{results['Error'].min():.2f}")

    st.subheader("Raw Dataset")
    st.dataframe(raw_df)


# Footer 
st.markdown("---")
st.caption("Built by Akash Anuragi | Air Passengers TSA | Linear Regression")
