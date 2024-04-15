from sklearn.linear_model import Perceptron
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

perceptron=Perceptron()
perceptron.fit(data_train,target_train)
target_pred=perceptron.predict(data_test)
accuracy=accuracy_score(target_test,target_pred)
print("Perceptron Accuracy:", accuracy)