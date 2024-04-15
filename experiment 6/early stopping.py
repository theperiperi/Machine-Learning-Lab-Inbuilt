from sklearn.neural_network import MLPClassifier
import pandas as pd 
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df=pd.read_csv('data.csv')
X = df.drop(["diagnosis"], axis=1)  # Features
y = df['diagnosis']  # Target

X = X.fillna(np.mean(X))
data_train,data_test,target_train,target_test=train_test_split(X,y,test_size=0.2,random_state=42)
scaler = StandardScaler()
data_train = scaler.fit_transform(data_train)
data_test = scaler.transform(data_test)

mlp=MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42,early_stopping=True)
mlp.fit(data_train,target_train)
target_pred=mlp.predict(data_test)
accuracy=accuracy_score(target_test,target_pred)
print("mlp Accuracy:", accuracy)