from fastapi import FastAPI, HTTPException,Request
import mlflow
from enum import Enum
from dotenv import load_dotenv
import os 
from src.serving.schema import build_request_model_from_signature
from fastapi import Body
import pandas as pd
from src.observability.middleware import request_id_middleware
from src.observability.logger import get_logger
from src.observability.metrics import Obj_Basic_Metrics
import time
from src.serving.lifecycle import IN_FLIGHT, IN_FLIGHT_LOCK

load_dotenv()
logger = get_logger("inference")

app = FastAPI(title="Deploying Credit risk Model",version="0.0.0.1")

app.middleware("http")(request_id_middleware)

# READY = False
MODEL = None
MODEL_INFO = None
REQUEST_MODEL = None

class Startupstate(str,Enum):
    Startup_starting = "Starting"
    Startup_Ready = "Ready"
    Starup_Failed = "Failed"
    SHUTTING_DOWN = "Shutting_Down"

Obj = Startupstate.Startup_starting

@app.get("/health")
def health():
    
    return {"Status" :"Ok","Start_State":Obj}


@app.get("/ready")
def ready():
    if Obj != Startupstate.Startup_Ready:
       raise HTTPException 
       {
           status_code : 503,
           "detail" : f"System Not Ready yet! Dont send the Traffic {Obj}"
       }
    return {"Startup_State" : Obj}






@app.on_event("startup")
def load_model():
    global  MODEL, MODEL_INFO,REQUEST_MODEL,Obj

    try:
        Obj = Startupstate.Startup_starting
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

        # READY = True
        Obj = Startupstate.Startup_Ready
        logger.error(
                    "startup_suceeed!!!!!!!!!!",
                    extra={
                        "yipeeee": Obj
                    },
                )

    except Exception as e:
        Obj = Startupstate.Starup_Failed
        READY = False
        logger.error(
                    "startup_failed",
                    extra={
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                    },
                )
        raise
@app.post("/predict")
def predict(request: Request, payload=Body(...)):
    if Obj == Startupstate.Starup_Failed:
        raise HTTPException(status_code=503, detail="Service not ready")

    try:
        validated = REQUEST_MODEL(**payload)
        df = pd.DataFrame([validated.dict()])
        
        probability = float(MODEL.predict(df)[0])
        prediction = int(probability >= 0.5)

        Obj_Basic_Metrics.inc_sucess()

        logger.info(
            "inference_success",
            extra={
                "request_id": request.state.request_id,
                "prediction": prediction,
                "probability": probability,
                "model_name": MODEL_INFO["name"],
                "model_alias": MODEL_INFO["alias"],
                "model_run_id": MODEL_INFO["run_id"],
            },
        )

        return {
            "prediction": prediction,
            "probability": probability,
        }

    except Exception as e:
        Obj_Basic_Metrics.inc_erros()
        logger.error(
            "inference_failure",
            extra={
                "request_id": request.state.request_id,
                "error_type": type(e).__name__,
                "error_message": str(e),
            },
        )
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/metrics/basic")
def expose_basic_metrics():
    return  Obj_Basic_Metrics.display_snapshot()
@app.on_event("shutdown")
def shutting_down():
    global Obj

    Obj = Startupstate.SHUTTING_DOWN
    logger.info("shutdown_started")

    timeout_s = 30
    start_time = time.time()

    while True:
        with IN_FLIGHT_LOCK:
            remaining = IN_FLIGHT

        if remaining == 0:
            break

        if time.time() - start_time > timeout_s:
            logger.warning(
                "shutdown_timeout",
                extra={"in_flight": remaining},
            )
            break

        time.sleep(0.1)

    logger.info("shutdown_complete")
