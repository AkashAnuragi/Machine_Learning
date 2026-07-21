import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Page Config
st.set_page_config(page_title="Iris DBSCAN Clustering", layout="wide")

st.title("Iris Dataset - DBSCAN Clustering")

# Load Dataset
df = pd.read_csv("dataset/Iris.csv")
df = df.drop(columns=["Id"])

st.subheader("Dataset")
st.dataframe(df, height=200)

# Select Features
X = df[['SepalLengthCm', 'SepalWidthCm',
        'PetalLengthCm', 'PetalWidthCm']]

# User Inputs
col1, col2 = st.columns(2)

with col1:
    eps = st.slider("EPS Value", 0.1, 2.0, 0.8, 0.1)

with col2:
    min_samples = st.slider("Min Samples", 2, 20, 5)

# Run DBSCAN
if st.button("Run DBSCAN"):

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    dbscan = DBSCAN(
        eps=eps,
        min_samples=min_samples
    )

    clusters = dbscan.fit_predict(X_scaled)

    df['Cluster'] = clusters

    st.subheader("Clustered Data")
    st.dataframe(df)

    st.write("Number of Clusters:", len(set(clusters)) - (1 if -1 in clusters else 0))
    st.write("Noise Points:", list(clusters).count(-1))

    # Plot
    fig, ax = plt.subplots(figsize=(6, 3))

    sns.scatterplot(
        x='PetalLengthCm',
        y='PetalWidthCm',
        hue='Cluster',
        palette='deep',
        data=df,
        ax=ax
    )

    ax.set_title("DBSCAN Clustering")
    st.pyplot(fig)