import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# dataset
solarData = pd.read_csv('solar_clean_data.csv')


# Features and target
X = solarData[['dust_value', 'voltage']].values
y = LabelEncoder().fit_transform(solarData['decision'])

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
Xtrain, Xtest, ytrain, ytest = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# logistic regression as 1-layer NN with sigmoid
model = Sequential([
    Dense(1, activation='sigmoid', input_shape=(2,))
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(Xtrain, ytrain, epochs=100, batch_size=16, verbose=0)

loss, acc = model.evaluate(Xtest, ytest, verbose=0)
print("TF Logistic Regression Accuracy:", acc)

# Save the model
model.export('logistic_model')  # saves a folder named 'logistic_model'

