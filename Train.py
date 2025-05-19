import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


solarData=pd.read_csv('solar_clean_data.csv')
# print(solarData)

X=solarData[['dust_value','voltage','current']]
y=solarData['decision']

Xtrain,Xtest,yTrain,yTest=train_test_split(X,y,test_size=0.2,random_state=42)