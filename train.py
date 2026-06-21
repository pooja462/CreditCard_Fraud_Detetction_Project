import pandas as pd
import joblib as jb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier


CONFIG = {
    "test_size": 0.2,
    "threshold": 0.2,
    "random_state": 42,
    "sampling_strategy":0.5
}
df=pd.read_csv(r'D:\fraud_detection\data\creditcard.csv')
X= df.drop("Class", axis=1)
y=df["Class"]
jb.dump(list(X.columns),"features.pkl")



X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=CONFIG["test_size"], stratify=y, random_state=CONFIG["random_state"]
)

# Handle imbalance
smote = SMOTE(sampling_strategy=CONFIG['sampling_strategy'], random_state=CONFIG['random_state'])
X_train, y_train = smote.fit_resample(X_train, y_train)
scaler_time=StandardScaler()
scaler_amount=StandardScaler()


X_train['Time']=scaler_time.fit_transform(X_train['Time'].values.reshape(-1,1))
X_train["Amount"]=scaler_amount.fit_transform(X_train['Amount'].values.reshape(-1,1))
jb.dump(scaler_time, "time_scaler.pkl")
jb.dump(scaler_amount, "amount_scaler.pkl")


# Train model
model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    scale_pos_weight=10
)
model.fit(X_train, y_train)


#evaluation
from sklearn.metrics import classification_report 
X_test["Time"] = scaler_time.transform(X_test["Time"].values.reshape(-1,1))
X_test["Amount"] = scaler_amount.transform(X_test["Amount"].values.reshape(-1,1))

probs=model.predict_proba(X_test)[:,1]
y_pred=(probs>CONFIG["threshold"]).astype(int)
print(classification_report(y_test,y_pred))
from sklearn.metrics import roc_auc_score
print(roc_auc_score(y_test, probs))
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred))

# Save model
jb.dump(model, "fraud_model.pkl")
print("Training complete")