from fastapi import FastAPI, HTTPException
import mlflow
from dotenv import load_dotenv
import os 
from src.serving.schema import build_request_model_from_signature
from fastapi import Body
import pandas as pd


load_dotenv()

app = FastAPI(title="Deploying Credit risk Model",version="0.0.0.1")
READY = False
MODEL = None
MODEL_INFO = None
REQUEST_MODEL = None


@app.get("/health")
def health():
    return {"Status" :"Ok"}


@app.get("/ready")
def ready():
    if not READY:
        raise HTTPException (status_code=503, detail="Not ready : Model Not Loaded")
    return {"ready":True}


@app.on_event("startup")
def load_model():
    global READY, MODEL, MODEL_INFO,REQUEST_MODEL

    try:
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

        model_name = "credit_risk_xgboost"
        model_alias = "champion"  

        model_uri = f"models:/{model_name}@{model_alias}"

        MODEL = mlflow.pyfunc.load_model(model_uri)

        MODEL_INFO = {
            "name": model_name,
            "alias": model_alias,
            "run_id": MODEL.metadata.run_id,
            "model_uri": model_uri,
        }



        signature = MODEL.metadata.signature
        REQUEST_MODEL = build_request_model_from_signature(signature)

        print("Request schema built successfully")

        READY = True
        print("Model loaded successfully")

    except Exception as e:
        READY = False
        print("Startup failed:", e)
        raise


@app.post("/predict")
def predict(payload=Body(...)):
    if not READY:
        raise HTTPException(status_code=503, detail="Service not ready")

    try:
        validated = REQUEST_MODEL(**payload)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"status": "validated"}
