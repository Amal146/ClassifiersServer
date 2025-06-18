from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import pickle
import os
from fastapi.middleware.cors import CORSMiddleware
import joblib

# ---------- Config ----------
BLOCKCHAIN_MODEL_PATH = r"C:\Users\amalj\OneDrive\Desktop\classifiers\app\models\blockchain_transaction_classifier.pkl"



if not os.path.exists(BLOCKCHAIN_MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {BLOCKCHAIN_MODEL_PATH}")
with open(BLOCKCHAIN_MODEL_PATH, "rb") as f:
    classifier = pickle.load(f)


script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "models", "xgb_model.pkl")
scaler_path = os.path.join(script_dir, "models", "scaler.pkl")

xgb_model = joblib.load(model_path)
scaler = joblib.load(scaler_path)


# ---------- FastAPI App ----------
app = FastAPI(
    title="Blockchain Classifier & Fraud Detection API",
    description="API for classifying Ethereum transactions and detecting fraud",
    version="1.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify your React app's URL e.g. ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Pydantic Models ----------
class TransactionFeatures(BaseModel):
    value_eth: float
    gas_cost_eth: float
    tx_fee_ratio: float
    is_contract_tx: int
    input_length: Optional[int] = None
    is_high_value: Optional[int] = None
    is_low_value: Optional[int] = None
    is_weekend: Optional[int] = None
    hour: Optional[int] = None
    day_of_week: Optional[int] = None
    month: Optional[int] = None
    time_of_day: Optional[str] = None
    is_defi: Optional[int] = None
    is_nft: Optional[int] = None
    protocol: Optional[str] = None


class PredictionResult(BaseModel):
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]




# Your full features list in order - must match model training
FEATURES_LIST = [
    'Avg min between sent tnx', 'Avg min between received tnx', 'Time Diff between first and last (Mins)',
    'Sent tnx', 'Received Tnx', 'Number of Created Contracts', 'Unique Received From Addresses',
    'Unique Sent To Addresses', 'min value received', 'max value received ', 'avg val received',
    'min val sent', 'max val sent', 'avg val sent', 'min value sent to contract', 'max val sent to contract',
    'avg value sent to contract', 'total transactions (including tnx to create contract', 'total Ether sent',
    'total ether received', 'total ether sent contracts', 'total ether balance', ' Total ERC20 tnxs',
    ' ERC20 total Ether received', ' ERC20 total ether sent', ' ERC20 total Ether sent contract',
    ' ERC20 uniq sent addr', ' ERC20 uniq rec addr', ' ERC20 uniq sent addr.1', ' ERC20 uniq rec contract addr',
    ' ERC20 avg time between sent tnx', ' ERC20 avg time between rec tnx', ' ERC20 avg time between rec 2 tnx',
    ' ERC20 avg time between contract tnx', ' ERC20 min val rec', ' ERC20 max val rec', ' ERC20 avg val rec',
    ' ERC20 min val sent', ' ERC20 max val sent', ' ERC20 avg val sent', ' ERC20 min val sent contract',
    ' ERC20 max val sent contract', ' ERC20 avg val sent contract', ' ERC20 uniq sent token name',
    ' ERC20 uniq rec token name'
]

class Transaction(BaseModel):
    blockNumber: str
    blockHash: str
    timeStamp: str
    hash: str
    nonce: str
    transactionIndex: str
    from_address: str = Field(..., alias="from")   # Accept 'from' key but use 'from_address' in model
    to_address: str = Field(..., alias="to")       # Same for 'to'
    value: str
    gas: str
    gasPrice: str
    input: str
    methodId: str
    functionName: str
    contractAddress: str
    cumulativeGasUsed: str
    txreceipt_status: str
    gasUsed: str
    confirmations: str
    isError: str

    # Use alias for JSON keys that clash with Python keywords
    class Config:
        fields = {
            'from_address': 'from'
        }

class TransactionsPayload(BaseModel):
    address: str
    transactions: List[Transaction]

def to_int(ts):
    try:
        return int(ts)
    except:
        return 0

def compute_features(transactions: List[Transaction], target_address: str) -> pd.DataFrame:
    target_address = target_address.lower()
    timestamps = [to_int(tx.timeStamp) for tx in transactions]
    timestamps_sorted = sorted(timestamps)

    feature_rows = []

    for tx in transactions:
        tx_time = to_int(tx.timeStamp)
        tx_value_eth = int(tx.value) / 1e18
        sent = (tx.from_address.lower() == target_address)
        received = (tx.to_address.lower() == target_address)

        previous_times = [t for t in timestamps_sorted if t < tx_time]
        next_times = [t for t in timestamps_sorted if t > tx_time]

        time_since_prev = (tx_time - max(previous_times)) / 60 if previous_times else 0
        time_to_next = (min(next_times) - tx_time) / 60 if next_times else 0

        features = {
            'Sent tnx': int(sent),
            'Received Tnx': int(received),
            'Time Diff between first and last (Mins)': (max(timestamps_sorted) - min(timestamps_sorted)) / 60 if len(timestamps_sorted) > 1 else 0,
            'Avg min between sent tnx': 0,   # You can implement avg logic here if needed
            'Avg min between received tnx': 0,
            'min val sent': tx_value_eth if sent else 0,
            'max val sent': tx_value_eth if sent else 0,
            'avg val sent': tx_value_eth if sent else 0,
            'min value received': tx_value_eth if received else 0,
            'max value received ': tx_value_eth if received else 0,
            'avg val received': tx_value_eth if received else 0,
        }

        # Fill missing features with 0
        for feat in FEATURES_LIST:
            if feat not in features:
                features[feat] = 0

        feature_rows.append(features)

    df = pd.DataFrame(feature_rows)
    df = df[FEATURES_LIST]  # reorder columns
    return df

# ---------- Blockchain Transaction Endpoints ----------
@app.post("/classify", response_model=PredictionResult)
async def predict_transaction(tx: TransactionFeatures):
    try:
        input_data = pd.DataFrame([tx.dict()])
        predicted_class = classifier.predict(input_data)[0]
        probabilities = classifier.predict_proba(input_data)[0]

        # Get class labels (as string)
        if hasattr(classifier, 'label_encoder'):
            class_labels = [str(x) for x in classifier.label_encoder.classes_]
        elif hasattr(classifier, 'classes_'): 
            class_labels = [str(x) for x in classifier.classes_]
        elif hasattr(classifier, 'named_steps') and 'classifier' in classifier.named_steps:
            class_labels = [str(x) for x in classifier.named_steps['classifier'].classes_]
        else:
            class_labels = [str(i) for i in range(len(probabilities))]

        proba_dict = {label: float(prob) for label, prob in zip(class_labels, probabilities)}

        return {
            "predicted_class": str(predicted_class),
            "confidence": float(np.max(probabilities)),
            "probabilities": proba_dict
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/batch_predict", response_model=List[PredictionResult])
async def batch_predict(transactions: List[TransactionFeatures]):
    try:
        input_data = pd.DataFrame([tx.dict() for tx in transactions])
        predictions = classifier.predict(input_data)
        probabilities = classifier.predict_proba(input_data)

        if hasattr(classifier, 'label_encoder'):
            class_labels = [str(x) for x in classifier.label_encoder.classes_]
        elif hasattr(classifier, 'classes_'):
            class_labels = [str(x) for x in classifier.classes_]
        elif hasattr(classifier, 'named_steps') and 'classifier' in classifier.named_steps:
            class_labels = [str(x) for x in classifier.named_steps['classifier'].classes_]
        else:
            class_labels = [str(i) for i in range(probabilities.shape[1])]

        results = []
        for pred, proba in zip(predictions, probabilities):
            results.append({
                "predicted_class": str(pred),
                "confidence": float(np.max(proba)),
                "probabilities": {
                    str(label): float(p)
                    for label, p in zip(class_labels, proba)
                }
            })

        return results

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/model_info")
async def get_model_info():
    if hasattr(classifier, 'label_encoder'):
        classes_str = ', '.join(map(str, classifier.label_encoder.classes_))
    elif hasattr(classifier, 'classes_'):
        classes_str = ', '.join(map(str, classifier.classes_))
    elif hasattr(classifier, 'named_steps') and 'classifier' in classifier.named_steps:
        classes_str = ', '.join(map(str, classifier.named_steps['classifier'].classes_))
    else:
        classes_str = "Unavailable (label_encoder missing)"
    return {
        "model_type": "XGBoost",
        "classes": classes_str,
        "best_params": getattr(classifier, 'best_params_', "N/A")
    }

# ---------- Fraud Detection Endpoints ----------
@app.post("/fraud_predict")
def predict(payload: TransactionsPayload):
    try:
        df_features = compute_features(payload.transactions, payload.address)
        scaled_features = scaler.transform(df_features)
        preds = xgb_model.predict(scaled_features)
        probs = xgb_model.predict_proba(scaled_features)

        results = []
        for tx, pred, prob in zip(payload.transactions, preds, probs):
            results.append({
                "tx_hash": tx.hash,
                "prediction": int(pred),
                "probability_class_0": float(prob[0]),
                "probability_class_1": float(prob[1]),
                "label": "🚨 Fraudulent" if pred == 1 else "✅ Not Fraudulent"
            })

        return {"results": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))