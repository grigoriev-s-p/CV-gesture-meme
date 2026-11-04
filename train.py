import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

def normalize(data):
    data = data.copy()

    points = data.iloc[:, :63].to_numpy().reshape(-1, 21, 3)
    points = points - points[:, 0:1, :]

    scale = np.linalg.norm(points[:, :, :2], axis=2).max(axis=1)
    scale[scale < 1e-6] = 1
    points = points / scale[:, None, None]
    data.iloc[:, :63] = points.reshape(-1, 63)

    return data

        
data = pd.read_csv("data.csv", header=None)
data = normalize(data)

X_train = data.iloc[:,:63]
y_train = data.iloc[:,63]


model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=1)

model.fit(X_train,y_train)

joblib.dump(model, "model.joblib")