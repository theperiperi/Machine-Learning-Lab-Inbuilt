import pandas as pd 
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

df=pd.read_csv('data.csv')
X=df.drop('diagnosis',axis=1)
y=df['diagnosis']

X=X.fillna(np.mean(X))
data_train,data_test,target_train,target_test=train_test_split(X,y,test_size=0.2,random_state=42)
scaler=StandardScaler()
data_train=scaler.fit_transform(data_train)
data_test=scaler.transform(data_test)

smote=SMOTE()
data_train,target_train=smote.fit_resample(data_train,target_train)

mlp=MLPClassifier()
mlp.fit(data_train,target_train)
prediction=mlp.predict(data_test)
accuracy=accuracy_score(target_test,prediction)

print(accuracy)