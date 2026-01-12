import uuid
import time
from fastapi import Request,HTTPException
from src.observability.metrics import Obj_Basic_Metrics
from src.observability.logger import get_logger


logger = get_logger()
async def request_id_middleware(request: Request, call_next):
    Obj_Basic_Metrics.inc_reuqest()


    request_id = str(uuid.uuid4())
    request.state.request_id = request_id

    start_time = time.perf_counter()
    try:
        response = await call_next(request)  # ← CRITICAL

    except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    latency_ms = (time.perf_counter() - start_time) * 1000
    request.state.latency_ms = latency_ms
    Obj_Basic_Metrics.record_latency(latency_ms)


    snapshot = Obj_Basic_Metrics.display_snapshot()

    logger.info(
        "request_completed_from_middleware",
        extra={
            "request_id": request.state.request_id,
            "latency_ms": round(latency_ms, 2),
            "Total_Request": snapshot["total_request"],
            "Total_Sucess": snapshot["total_sucess"],
            "Total_Failures": snapshot["total_erros"],
            "Error_Rate":snapshot["error_rate"],
            "path": request.url.path,
            "method": request.method,
        },
    )



    response.headers["X-Request-ID"] = request_id
    response.headers["X-Latency-ms"] = f"{latency_ms:.2f}"

    return response
