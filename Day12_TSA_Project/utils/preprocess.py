import joblib
import numpy as np

from sklearn.preprocessing import MinMaxScaler

class DataPreprocessor:
    def __init__(self,time_step=60):
        self.time_step = time_step
        self.scaler = MinMaxScaler(feature_range=(0,1))

    def preprocess(self,df):
        # keep only 'Close' price
        data = df[['Close']].copy()
        # Handle missing values
        data = data.dropna()
        scaled_data = self.scaler.fit_transform(data)
        X = []
        y = []
        # Create Sequences
        for i in range(self.time_step , len(scaled_data)):
            X.append(scaled_data[i-self.time_step : i , 0])
            y.append(scaled_data[i,0])
        X = np.array(X)
        y = np.array(y)
        # Reshape for Model (LSTM)
        X = X.reshape((X.shape[0], X.shape[1],1))
        return X,y
    
    def save_scaler(self , path="models/scaler.pkl"):
        joblib.dump(self.scaler , path)

    def load_scaler(self , path="models/scaler.pkl"):
        self.scaler = joblib.load(path)

    def inverse_transform(self,values):
        return self.scaler.inverse_transform(values)
    