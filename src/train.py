import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
from src.collect import collect_data_for_training


def normalize(data):
    data = data.copy()

    points = data.iloc[:, :63].to_numpy().reshape(-1, 21, 3)
    points = points - points[:, 0:1, :]

    scale = np.linalg.norm(points[:, :, :2], axis=2).max(axis=1)
    scale[scale < 1e-6] = 1
    points = points / scale[:, None, None]
    data.iloc[:, :63] = points.reshape(-1, 63)

    return data

class Model:
    def __init__(self,n_est=200, random_state=42, n_jobs=1):
        self.model =  RandomForestClassifier(n_estimators=n_est, random_state=random_state, n_jobs=n_jobs)
    
    def fit(self,collect_data = False,data = None, path = None, name_for_model_file = "model"):
        if data == None and collect_data == False:
            try:
                self.data = pd.read_csv(path, header = None)
            except Exception as e:
                print(e)
        if collect_data:
            data = collect_data_for_training()
        self.data = normalize(data)

        X_train = data.iloc[:,:63]
        y_train = data.iloc[:,63]
        self.fit = self.model.fit(X_train,y_train)
        joblib.dump(self.model, f"{name_for_model_file}.joblib")

model = Model()
model.fit(path = "C:/Users/11211/Desktop/github/pet-projec/data/data.csv")
