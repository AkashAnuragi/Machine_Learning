from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import LSTM , Dense , Dropout
from utils.config import TIME_STEP

def build_lstm_model(time_step = TIME_STEP):
    model = Sequential()
    model.add(
        LSTM(64 , return_sequences=True , input_shape=(time_step,1))
    )
    model.add(Dropout(0.2))
    model.add(LSTM(64))
    model.add(Dropout(0.2))
    model.add(Dense(32,activation='relu'))
    model.add(Dense(1))

    model.compile(
        optimizer='adam',
        loss="mean_squared_error",
        metrics=["mae"]
    )
    return model