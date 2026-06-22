import pandas as pd 
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Loading Data
df = pd.read_csv("dataset/Iris.csv")

df = df.drop(columns=['Id'])

# Removing missing Values
df = df.dropna()

# Setting UI Page
st.set_page_config(page_title="Iris Classification", layout='wide' )
st.title("Iris Classification Model")
st.text("We can predict iris species here like Sentosa, Versicolor and Verginca, you need to input sepal length and width and as well as petal length and width.")
st.dataframe(df, height=200)

# Plotting Actuall Data
fig, ax = plt.subplots (1,2,figsize=(12,3))
sns.scatterplot(x='SepalLengthCm' , y = 'SepalWidthCm', hue = 'Species' , data = df ,  ax = ax[0])
sns.scatterplot(x='PetalLengthCm' , y='PetalWidthCm' , hue='Species' , data=df , ax=ax[1])
st.pyplot(fig)

col1,col2 = st.columns(2)
with col1:
    sl = st.number_input("Enter Sepal Length in cm")
    sw = st.number_input("Enter Sepal Width in cm")
with col2:
    pl = st.number_input("Enter Petal Length in cm")
    pw = st.number_input("Enter Petal Width in cm")

# Loading Model
import joblib
model = joblib.load("model/model.pkl")

if st.button("Predict"):
    data = np.array([[sl,sw,pl,pw]])
    pred = model.predict(data)
    li = ['Setosa','Versicolor','Verginica']
    st.success(li[pred[0]])
