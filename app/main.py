from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import pickle
import os
from fastapi.middleware.cors import CORSMiddleware


# ---------- Config ----------
BLOCKCHAIN_MODEL_PATH = r"C:\Users\amalj\OneDrive\Desktop\classifiers\app\models\blockchain_transaction_classifier.pkl"
FRAUD_MODEL_PATH = r"C:\Users\amalj\OneDrive\Desktop\classifiers\app\models\lstm_model.pkl"
LSTM_SCALER_PATH = r"C:\Users\amalj\OneDrive\Desktop\classifiers\app\models\lstm_scaler.pkl"


if not os.path.exists(BLOCKCHAIN_MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {BLOCKCHAIN_MODEL_PATH}")
if not os.path.exists(FRAUD_MODEL_PATH):
    raise FileNotFoundError(f"Fraud model not found at {FRAUD_MODEL_PATH}")

with open(BLOCKCHAIN_MODEL_PATH, "rb") as f:
    classifier = pickle.load(f)
with open(FRAUD_MODEL_PATH, "rb") as f:
    fraud_model = pickle.load(f)
with open(LSTM_SCALER_PATH, "rb") as f:
    lstm_scaler = pickle.load(f)

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


class FraudDetectionFeatures(BaseModel):
    # Match exactly what the scaler expects
    Sent_tnx: int = Field(..., alias="Sent tnx")
    Received_Tnx: int = Field(..., alias="Received Tnx")
    Unique_Received_From_Addresses: int = Field(..., alias="Unique Received From Addresses")
    Unique_Sent_To_Addresses: int = Field(..., alias="Unique Sent To Addresses")
    total_ether_received: float = Field(..., alias="total ether received")
    avg_val_received: float = Field(..., alias="avg val received")
    avg_val_sent: float = Field(..., alias="avg val sent")
    min_value_received: float = Field(..., alias="min value received")
    max_value_received: float = Field(..., alias="max value received")
    min_val_sent: float = Field(..., alias="min val sent")
    max_val_sent: float = Field(..., alias="max val sent")
    total_Ether_sent: float = Field(..., alias="total Ether sent")
    Time_Diff_between_first_and_last_Mins: float = Field(..., alias="Time Diff between first and last (Mins)")

    class Config:
        allow_population_by_field_name = True
        extra = "forbid"

class FraudResult(BaseModel):
    predicted_label: str
    confidence: float
    probabilities: Dict[str, float]


# ---------- Blockchain Transaction Endpoints ----------
@app.post("/predict", response_model=PredictionResult)
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

@app.post("/fraud_predict", response_model=FraudResult)
async def fraud_predict(features: FraudDetectionFeatures):
    try:
        # Convert to dict with original feature names
        input_data = features.dict(by_alias=True)
        
        # Create DataFrame with ALL original features (zero-filled for missing ones)
        full_features = {name: 0 for name in lstm_scaler.feature_names_in_}
        full_features.update(input_data)  # Override with our provided values
        input_df = pd.DataFrame([full_features])[lstm_scaler.feature_names_in_]
        
        # Scale and predict
        scaled_input = lstm_scaler.transform(input_df)
        X_input = scaled_input.reshape((1, 1, scaled_input.shape[1]))
        pred = fraud_model.predict(X_input)[0][0]

        return {
            "predicted_label": '1' if pred >= 0.5 else '0',
            "confidence": float(max(pred, 1 - pred)),
            "probabilities": {'0': 1 - pred, '1': pred}
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    

@app.post("/fraud_batch_predict", response_model=List[FraudResult])
async def fraud_batch_predict(features_batch: List[FraudDetectionFeatures]):
    try:
        df_batch = pd.DataFrame([f.dict() for f in features_batch])
        scaled_batch = lstm_scaler.transform(df_batch)
        X_batch = scaled_batch.reshape((len(df_batch), 1, df_batch.shape[1]))

        preds = fraud_model.predict(X_batch).flatten()

        results = []
        for p in preds:
            label = '1' if p >= 0.5 else '0'
            proba_dict = {'0': 1 - p, '1': p}
            results.append({
                "predicted_label": label,
                "confidence": float(max(p, 1 - p)),
                "probabilities": proba_dict
            })

        return results
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
