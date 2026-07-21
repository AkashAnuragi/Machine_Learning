import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')   # This line is needed to save plots without opening a window
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib   # joblib is used to save and load the trained model


# Load the Dataset

df = pd.read_csv("dataset/AirPassengers.csv",header=0, names= ['Month','Passengers'])

# Convert month column to date format
df['Month'] = pd.to_datetime(df['Month'])

# Set Month as the index 
df.set_index('Month', inplace = True)

print("Dataset Loaded")
print(f"   Shape: {df.shape}")          # (144, 1) means 144 rows, 1 column
print(f"   First row: {df.index[0]}")
print(f"   Last row:  {df.index[-1]}")
print()


df['Lag1'] = df['Passengers'].shift(1)
df['Lag2'] = df['Passengers'].shift(2)
df['Lag3'] = df['Passengers'].shift(3)
df['Lag6'] = df['Passengers'].shift(6)
df['Lag12'] = df['Passengers'].shift(12)

df.dropna(inplace= True)

print("Lag Features Created!")
print(f"   Rows after dropping NaN: {len(df)}")
print()

X = df[['Lag1', 'Lag2', 'Lag3', 'Lag6', 'Lag12']]
y = df['Passengers']


train_size = int(len(df)*0.80)  

X_train = X.iloc[:train_size] # first 80%
X_test = X.iloc[train_size:]  # last 20%

y_train = y.iloc[:train_size] 
y_test = y.iloc[train_size:] 

print("Train-Test Split Done!")
print(f"Train size: {len(X_train)} rows ({X_train.index[0].date()} -> {X_train.index[-1].date()})")
print(f"Test size: {len(X_test)} rows ({X_test.index[0].date()} -> {X_test.index[-1].date()})")


# Train Model
# input +output --> mx+c
model = LinearRegression()
model.fit(X_train, y_train) 
print( "Model trained")
print()


# Make Prediction

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test,predictions)
rmse = np.sqrt(mean_absolute_error(y_test,predictions))

print("Model Results: ")
print(f"   MAE  = {mae:.2f}  (on average, prediction is off by {mae:.0f} passengers)")
print(f"   RMSE = {rmse:.2f}")
print()

# Save the plot 
fig ,ax = plt.subplots(figsize=(12,3))

# plot actual value
ax.plot(y_test.index,y_test.values,
        label= 'Actual',
        color='steelblue',
        linewidth=2
        )

# Predicted values (dash line)
ax.plot(y_test.index,predictions,
        label=f"Predicted (MAE = {mae:.1f})",
        color='orange',
        linewidth =2,
        linestyle='--'
        )

ax.set_title("Air Passengers -- Actual vs Predicted", fontsize=14),
ax.set_xlabel('Month')
ax.set_ylabel('Passengers (thousands)')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/plots/forecast.png', dpi=150)
plt.close()

print("Plot saved")
print()

# Save the Trained Model and Results

joblib.dump(model,'model/lr_model.pkl')

# Save the actual vs Predicted table as a CSV file

result_df = pd.DataFrame({
    'Month' : y_test.index,
    'Actual' : y_test.values,
    'Predicted' : predictions.round(2),
    'Error' : (y_test.values - predictions).round(2)
})

result_df.to_csv('model/results.csv', index=False)

with open('model/metrics.txt','w') as f:
    f.write(f"MAE={mae:.2f}\n")
    f.write(f"RMSE={rmse:.2f}\n")

print("Model saved  -> model/lr_model.pkl")
print("Results saved -> model/results.csv")
print("Metrics saved -> model/metrics.txt")
print()

print("  Training Complete!")
