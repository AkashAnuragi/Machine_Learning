import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder


# Load Data
df = pd.read_csv("dataset\Churn_Modelling.csv")

# Drops Columns that don't  Help Prediction (Id,name)

df_model  =df.drop(columns=["RowNumber", "CustomerId", "Surname"])


# Encode Categorical columns
le_gender = LabelEncoder()
df["Gender"] = le_gender.fit_transform(df["Gender"])  # Female=0, Male=1
 
# One-hot encode Geography (drop_first to avoid redundancy)
df = pd.get_dummies(df, columns=["Geography"], drop_first=True)
