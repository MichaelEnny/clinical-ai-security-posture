"""
Clinical AI Inference API
FastAPI application with JWT-based RBAC and append-only audit logging.
"""

import os
import json
import hashlib
import datetime
import pickle
from typing import Optional

from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import jwt

from auth import (
    authenticate_user, create_access_token, get_current_user,
    require_role, USERS_DB
)
from audit import write_audit_log, get_audit_logs

app = FastAPI(title="Clinical AI Inference API", version="1.0")

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
LOADED_MODELS = {}


def load_model(name: str):
    if name not in LOADED_MODELS:
        path = os.path.join(MODEL_DIR, f"{name}_rf.pkl")
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail=f"Model {name} not found")
        with open(path, "rb") as f:
            LOADED_MODELS[name] = pickle.load(f)
        meta_path = os.path.join(MODEL_DIR, f"{name}_meta.json")
        with open(meta_path) as f:
            LOADED_MODELS[f"{name}_meta"] = json.load(f)
    return LOADED_MODELS[name], LOADED_MODELS[f"{name}_meta"]


class PredictionRequest(BaseModel):
    patient_id: str
    model_name: str
    features: dict


class PredictionResponse(BaseModel):
    patient_id_hash: str
    model_name: str
    model_version: str
    prediction: int
    confidence: float
    timestamp: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str


@app.post("/auth/token", response_model=TokenResponse)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
        )
    token = create_access_token(
        data={"sub": user["username"], "role": user["role"]}
    )
    return {"access_token": token, "token_type": "bearer"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: Request,
    body: PredictionRequest,
    current_user: dict = Depends(require_role(["clinician", "admin"]))
):
    model, meta = load_model(body.model_name)
    feature_names = meta["feature_names"]

    missing = [f for f in feature_names if f not in body.features]
    if missing:
        write_audit_log(
            event="PREDICTION_FAILED",
            user_id=current_user["username"],
            role=current_user["role"],
            patient_id_hash=hashlib.sha256(body.patient_id.encode()).hexdigest()[:16],
            model_name=body.model_name,
            model_version=meta["model_version"],
            input_hash="",
            prediction=None,
            confidence=None,
            detail=f"Missing features: {missing}",
        )
        raise HTTPException(
            status_code=422,
            detail=f"Missing features: {missing}"
        )

    feature_values = [body.features[f] for f in feature_names]
    input_hash = hashlib.sha256(
        json.dumps(feature_values, sort_keys=True).encode()
    ).hexdigest()[:16]

    prediction = int(model.predict([feature_values])[0])
    confidence = float(model.predict_proba([feature_values])[0][prediction])
    patient_id_hash = hashlib.sha256(body.patient_id.encode()).hexdigest()[:16]
    timestamp = datetime.datetime.utcnow().isoformat()

    write_audit_log(
        event="PREDICTION",
        user_id=current_user["username"],
        role=current_user["role"],
        patient_id_hash=patient_id_hash,
        model_name=body.model_name,
        model_version=meta["model_version"],
        input_hash=input_hash,
        prediction=prediction,
        confidence=round(confidence, 4),
        detail="",
    )

    return PredictionResponse(
        patient_id_hash=patient_id_hash,
        model_name=body.model_name,
        model_version=meta["model_version"],
        prediction=prediction,
        confidence=round(confidence, 4),
        timestamp=timestamp,
    )


@app.get("/predictions/history")
async def prediction_history(
    current_user: dict = Depends(require_role(["clinician", "admin"]))
):
    logs = get_audit_logs()
    if current_user["role"] == "clinician":
        logs = [l for l in logs if l.get("user_id") == current_user["username"]]
    return {"predictions": logs}


@app.get("/admin/users")
async def list_users(
    current_user: dict = Depends(require_role(["admin"]))
):
    return {"users": [
        {"username": u, "role": v["role"]}
        for u, v in USERS_DB.items()
    ]}


@app.get("/audit/logs")
async def audit_logs(
    current_user: dict = Depends(require_role(["auditor", "admin"]))
):
    return {"logs": get_audit_logs()}


@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0"}
