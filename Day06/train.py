import pandas as pd 

# Loadin  Data
df = pd.read_csv("dataset/Iris.csv")
df = df.drop(columns=['Id'])

# Removing Missing Values
df = df.dropna()   

# Data Encoding
df['Species'] = df['Species'] .map(lambda val: list(df.Species.unique()).index(val))

# Data Split in Features and Labels
X = df.drop(columns=['Species'])
y = df['Species']

# Data Split in Train - Test  
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2 , random_state=42)

# Model Selection
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()


# Modeling Training
model.fit(X_train,y_train)

# Model Prediction
y_pred = model.predict(X_test)


# Model Evaluation
from sklearn.metrics import accuracy_score , classification_report , confusion_matrix
acc = accuracy_score(y_test,y_pred)
cm = confusion_matrix(y_test,y_pred)
cr = classification_report(y_test,y_pred)
print("Accuracy Score :",acc*100)

# Save Model 
import joblib # joblib or pickle both are same
joblib.dump( model , "model/model.pkl" )

# Save the model evaluation
file = open("evaluation.txt",'a')
data = str(acc*100) + "\n" + str(cm) + '\n' + str(cr)
file.write( data )
file.close()

print("Model Training and Evaluation is completed!") 