import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# dataset load karo
df = pd.read_csv(r"D:\fraud_detection\data\creditcard.csv")

# ONLY single prediction features
X = df[['Time', 'Amount']]
y = df['Class']

# split (good practice)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# model train
model = LogisticRegression()
model.fit(X_train, y_train)

# save model
joblib.dump(model, "single_model.pkl")

print("Single model trained successfully!")