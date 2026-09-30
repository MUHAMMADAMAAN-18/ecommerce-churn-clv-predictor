import os, sys, pickle
sys.path.append(os.path.dirname(__file__))
from features import load_and_process
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "online_retail.csv")

print(f"Loading from {DATA_PATH}")
df = load_and_process(DATA_PATH)
print(f"Customers: {len(df)}, Churn rate: {df['Churn'].mean():.2f}")

X = df[['Recency', 'Frequency', 'Monetary_Log', 'AvgBasketValue']]
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, eval_metric='logloss')
model.fit(X_train, y_train)

preds = model.predict(X_test)
print(classification_report(y_test, preds))
print("ROC-AUC:", roc_auc_score(y_test, model.predict_proba(X_test)[:,1]))

os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
with open(os.path.join(BASE_DIR, "models", "churn_model.pkl"), "wb") as f:
    pickle.dump(model, f)
print("Model saved to models/churn_model.pkl - MashAllah!")