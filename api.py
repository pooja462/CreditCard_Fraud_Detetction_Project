from fastapi import FastAPI, UploadFile, File, Header, HTTPException, Depends
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from jose import jwt, JWTError, ExpiredSignatureError
from datetime import datetime, timedelta, timezone
from passlib.context import CryptContext
import pandas as pd
import joblib as jb
import os
import traceback
from db import connected, cursor
#for frontend
from fastapi.middleware.cors import CORSMiddleware


def safe_query(query, params):
    try:
        cursor.execute(query, params)
        connected.commit()
    except Exception as e:
        connected.rollback()
        raise e
#new code
single_model = jb.load("single_model.pkl")

app = FastAPI()

#for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# SECURITY CONFIG
SECRET_KEY = os.getenv("SECRET_KEY", "dev_secret_key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], bcrypt__rounds=12)


# ML MODELS
model = None
scaler_time = None
scaler_amount = None
features = None


@app.on_event("startup")
def load_artifacts():
    global model, scaler_time, scaler_amount, features
    model = jb.load("fraud_model.pkl")
    scaler_time = jb.load("time_scaler.pkl")
    scaler_amount = jb.load("amount_scaler.pkl")
    features = jb.load("features.pkl")



# SCHEMAS
class Transaction(BaseModel):
    Time: float
    V1: float; V2: float; V3: float; V4: float; V5: float
    V6: float; V7: float; V8: float; V9: float; V10: float
    V11: float; V12: float; V13: float; V14: float; V15: float
    V16: float; V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float; V25: float
    V26: float; V27: float; V28: float
    Amount: float


class UserCreate(BaseModel):
    username: str
    email: str
    password: str


class LoginData(BaseModel):
    username: str
    password: str



# JWT
def create_token(data: dict):
    payload = data.copy()
    payload["exp"] = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def verify_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload.get("user_id")
    except ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")


def get_user_id(authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid token format")
    token = authorization.split(" ")[1]
    return verify_token(token)


# CREATE USER
@app.post("/create-user")
def create_user(user: UserCreate):

    hashed_password = pwd_context.hash(user.password)

    cursor.execute("""
        INSERT INTO users(username, email, password)
        VALUES (%s, %s, %s)
        RETURNING id
    """, (user.username, user.email, hashed_password))

    user_id = cursor.fetchone()[0]
    connected.commit()

    return {"success": True, "user_id": user_id}


# LOGIN
@app.post("/login")
def login(data: LoginData):

    cursor.execute(
        "SELECT id, password FROM users WHERE username=%s",
        (data.username,)
    )
    user = cursor.fetchone()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    user_id, db_password = user

    if not pwd_context.verify(data.password, db_password):
        raise HTTPException(status_code=401, detail="Wrong password")

    token = create_token({"user_id": user_id})

    return {"success": True, "access_token": token}

#single prediction
from fastapi import Form, HTTPException

@app.post("/predict_single")
def predict_single(
    Time: float = Form(...),
    Amount: float = Form(...),
    user_id: int = Depends(get_user_id)
):

    prediction = single_model.predict([[Time, Amount]])[0]

    result_text = "Fraud" if prediction == 1 else "Not Fraud"

    probability = 0.0  # (optional: agar model prob nahi deta to 0)

    risk_level = "High" if prediction == 1 else "Low"

    # SAVE TO DATABASE
    cursor.execute("""
        INSERT INTO predictions (user_id, probability, result, risk_level)
        VALUES (%s, %s, %s, %s)
    """, (user_id, float(probability), int(prediction), risk_level))

    connected.commit()

    return {
        "success": True,
        "result": result_text,
        "risk_level":risk_level
    }

# BULK PREDICTION
@app.post("/predict-bulk")
def predict_bulk(file: UploadFile = File(...), user_id: int = Depends(get_user_id)):

    try:
        df = pd.read_csv(file.file)
        df = df.reindex(columns=features, fill_value=0)

        df["Time"] = scaler_time.transform(df[["Time"]])
        df["Amount"] = scaler_amount.transform(df[["Amount"]])

        probs = model.predict_proba(df)[:, 1]
        predictions = (probs > 0.5).astype(int)

        risk_levels = [
            "Low" if p <= 0.5 else "Medium" if p <= 0.8 else "High"
            for p in probs
        ]

        df["probability"] = probs
        df["prediction"] = predictions
        df["risk_level"] = risk_levels

        frauds = df[df["prediction"] == 1]

        cursor.executemany("""
            INSERT INTO bulk_predictions (user_id, probability, prediction, risk_level)
            VALUES (%s, %s, %s, %s)
        """, [
            (user_id, float(probs[i]), int(predictions[i]), risk_levels[i])
            for i in range(len(df))
        ])

        connected.commit()

        return {
            "success": True,
            "total": len(df),
            "fraud_count": len(frauds),
            "fraud_cases": frauds.to_dict(orient="records")
        }
    except Exception as e:
           connected.rollback()
           print("BULK ERROR:", str(e))  
           raise HTTPException(status_code=500, detail=str(e))
# HISTORY API (FIXED)
@app.get("/history")
def history(
    user_id: int = Depends(get_user_id),
    type: str | None = None,
    only_fraud: bool = False
):
    # SINGLE HISTORY
    if type == "single":

        cursor.execute("""
            SELECT 
                p.id,
                p.probability,
                p.result,
                p.risk_level,
                p.created_at,
                u.username
            FROM predictions p
            JOIN users u ON p.user_id = u.id
            WHERE p.user_id = %s
            ORDER BY p.id DESC
        """, (user_id,))

        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        data = [dict(zip(columns, row)) for row in rows]

        for d in data:
            if isinstance(d.get("result"), str):
                d["result"] = 1 if d["result"].lower() == "fraud" else 0

        if only_fraud:
            data = [d for d in data if int(d.get("result")) == 1]

        return {
            "success": True,
            "type": "single",
            "total": len(data),
            "fraud_count": sum(1 for d in data if int(d.get("result")) == 1),
            "data": data
        }
    # BULK HISTORY
    elif type == "bulk":

        cursor.execute("""
            SELECT 
                b.id,
                b.probability,
                b.prediction,
                b.risk_level,
                b.created_at,
                u.username
            FROM bulk_predictions b
            JOIN users u ON b.user_id = u.id
            WHERE b.user_id = %s
            ORDER BY b.id DESC
        """, (user_id,))

        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        data = [dict(zip(columns, row)) for row in rows]

        if only_fraud:
            data = [d for d in data if d.get("prediction") == 1]

        return {
            "success": True,
            "type": "bulk",
            "total": len(data),
            "fraud_count": sum(1 for d in data if d.get("prediction") == 1),
            "normal_count": sum(1 for d in data if d.get("prediction") == 0),
            "fraud_records": [d for d in data if d.get("prediction") == 1]
        }
# ERROR HANDLER
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):

    missing_fields = [err["loc"][-1] for err in exc.errors() if err["type"] == "missing"]

    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "message": "Missing required fields",
            "missing_fields": missing_fields
        }
    )