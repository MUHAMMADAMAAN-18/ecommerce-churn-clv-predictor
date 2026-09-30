from fastapi import FastAPI
import pickle
import pandas as pd

app = FastAPI()
model = pickle.load(open("models/churn_model.pkl","rb"))

@app.get("/")
def home():
    return {"message": "Churn API Running - MashAllah"}

@app.post("/predict")
def predict(Recency: int, Frequency: int, Monetary_Log: float, AvgBasketValue: float):
    data = [[Recency, Frequency, Monetary_Log, AvgBasketValue]]
    # Note: Order = Recency, Frequency, Monetary_Log, AvgBasketValue
    pred = model.predict(data)[0]
    prob = model.predict_proba(data)[0][1]
    return {"churn": int(pred), "churn_probability": float(prob)}