import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler


solarData=pd.read_csv('solar_clean_data.csv')
# print(solarData)

# selecting columns for training
X=solarData[['dust_value','voltage','current']]

# converting decision to numerics

enc=LabelEncoder()
y=enc.fit_transform(solarData['decision'])




# splitting data
Xtrain,Xtest,yTrain,yTest=train_test_split(X,y,test_size=0.2,random_state=42)

# scaling
scaler = StandardScaler()
Xtrain_scaled = scaler.fit_transform(Xtrain)
Xtest_scaled = scaler.transform(Xtest)

# training model
logistic=LogisticRegression()
logistic.fit(Xtrain_scaled, yTrain)

#evaluation
y_pred = logistic.predict(Xtest_scaled)
print("Accuracy:", accuracy_score(yTest, y_pred))
print(classification_report(yTest, y_pred))