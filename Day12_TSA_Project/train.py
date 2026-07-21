from sklearn.model_selection import train_test_split
from datetime import date
import os

from utils.model import build_lstm_model
from utils.preprocess import DataPreprocessor
from utils.data_loader import download_stock_data

from utils.config import *

def train_model(symbol,start,end,X,y):
    os.makedirs("models" , exist_ok=True)

    # Split Data
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=TEST_SIZE, shuffle=False)

    # Build Model
    model = build_lstm_model()

    # Train The Model
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test,y_test),
        epochs = EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1
    )

    # Save Model
    model.save(MODEL_PATH)
    print("Training Complete")