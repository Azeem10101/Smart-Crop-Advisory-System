import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("Crop_recommendation.csv")

# Features & labels
X = df.drop("Crop", axis=1)
y = df["Crop"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Accuracy
pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))

# Test prediction
sample = [[90, 40, 40, 20, 80, 6.5, 200]]
print("Prediction:", model.predict(sample))

import pickle

pickle.dump(model, open("model.pkl", "wb"))