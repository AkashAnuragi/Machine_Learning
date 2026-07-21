from preprocessing.load_data import dataset
from preprocessing.train import get_model
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# page SETUP
st.set_page_config(page_title="Customer Segmentation", layout = 'wide')

#load Dataset
df = dataset()

# HEADER
st.title("Customer Segmentation Project")
#st.text("This project is related to the customer segmentation and sales analysis with KPIs , Slicers and visualization")
st.write(
    "This project is related to Customer Segmentation using DBSCAN clustering "
    "along with Sales Analysis."
)

st.header("Total Number of Sales : "+str(df.dropna().shape[0]))
st.subheader("Total Number of Customer : "+str(len(list(df.CID.unique()))))



# DataFrame's Copy
filtered_df = df.copy() 

# Slicers
col1 , col2 , col3 , col4 , col5 = st.columns(5)

with col1:
    cid = st.selectbox("Select CustomerID",['All']+ list(filtered_df.CID.unique()))
    if cid!='All':
        filtered_df = filtered_df[filtered_df.CID == cid]

with col2:
    cname = st.selectbox("Select Customer Name",['All']+list(filtered_df.Cname.unique()))
    if cname!='All':
        filtered_df = filtered_df[filtered_df.Cname == cname]

with col3:
    state = st.selectbox("Select Customer State",['All']+list(filtered_df.State.unique()))

    if state != 'All':
        filtered_df = filtered_df[filtered_df.State == state]

with col4:
    city = st.selectbox("Select Customer City",['All']+list(filtered_df.City.unique()))

    if city != 'All':
        filtered_df = filtered_df[filtered_df.City == city]

with col5:
    gender = st.selectbox("Select Customer Gender",['All']+list(filtered_df.Gender.unique()))

    if gender != 'All':
        filtered_df = filtered_df[filtered_df.Gender == gender]


col1,col2,col3,col4,col5,col6 = st.columns([1.2,1.2,1.1,1.1,0.9,0.7])
with col1:
    pcat = st.selectbox("Select Product Category",['All']+list(filtered_df['Product Category'].unique()))   
    if pcat!='All':
        filtered_df = filtered_df[filtered_df['Product Category']==pcat] 
with col2:
    if (list(filtered_df['Product Sub-Category'].unique())) is None:
        pscat = st.selectbox("Select Product Sub-Category",['All'])      
    else:
        pscat = st.selectbox("Select Product Sub-Category",['All']+list(filtered_df['Product Sub-Category'].unique()))   
    if pscat!='All':
        filtered_df = filtered_df[filtered_df['Product Sub-Category']==pscat] 
with col3:
    st.metric("Total Sales",float(filtered_df.Amt.sum()))
with col4:
    st.metric("Total Profit",float(filtered_df.Profit.sum()))
with col5:
    st.metric("Total Sold Quantity",float(filtered_df.Qty.sum()))
with col6:
    st.metric("Total No of Sales",int(filtered_df.dropna().shape[0]))

# Visualization
col1 , col2 = st.columns(2)
with col1:
    fig , ax = plt.subplots( figsize=(6,2.9) )
    pbpc = filtered_df.dropna().groupby("Product Category")['Profit'].sum().reset_index()
    ax.bar( pbpc['Product Category'] , pbpc['Profit'] )
    st.pyplot(fig)
with col2:
    fig , ax = plt.subplots( figsize=(12,7) )
    pbpc = filtered_df.dropna().groupby("Product Sub-Category")['Profit'].sum().reset_index()
    ax.barh( pbpc['Product Sub-Category'] , pbpc['Profit'] )
    st.pyplot(fig)

st.subheader("Filtered Dataset")
st.dataframe(filtered_df,height=200)

st.subheader("DBSCAN Parameters")

col1, col2 = st.columns(2)

with col1:
    eps = st.slider(
        "Epsilon (eps)",
        min_value=0.1,
        max_value=5.0,
        value=1.0,
        step=0.1
    )

with col2:
    min_samples = st.slider(
        "Minimum Samples",
        min_value=2,
        max_value=20,
        value=4,
        step=1
    )

 
labels = get_model(
    df,
    eps=eps,
    min_samples=min_samples
)

# ---------------- Cluster Information ----------------

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

noise_points = np.sum(labels == -1)

col1, col2 = st.columns(2)

with col1:
    st.metric("Clusters Found", n_clusters)

with col2:
    st.metric("Noise Points", noise_points)


# ---------------- Scatter Plot ----------------

st.subheader("Customer Segmentation (DBSCAN)")

fig, ax = plt.subplots(figsize=(12, 3))

clean_df = df.dropna()

scatter = ax.scatter(
    clean_df["Profit"],
    clean_df["Qty"],
    c=labels,
    cmap="viridis",
    s=60
)

ax.set_xlabel("Profit")
ax.set_ylabel("Quantity")
ax.set_title("DBSCAN Clustering")

plt.colorbar(scatter)

st.pyplot(fig)



