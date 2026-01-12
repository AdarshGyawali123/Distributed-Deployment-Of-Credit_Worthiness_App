import logging
import json
import sys
from datetime import datetime


class JsonFormatter(logging.Formatter):
    def format(self, record):
        log = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "event": record.getMessage(),
        }

        # Standard LogRecord attributes we should ignore
        reserved = {
            "name", "msg", "args", "levelname", "levelno",
            "pathname", "filename", "module", "exc_info",
            "exc_text", "stack_info", "lineno", "funcName",
            "created", "msecs", "relativeCreated", "thread",
            "threadName", "processName", "process",
        }

        # Extract custom structured fields
        for key, value in record.__dict__.items():
            if key not in reserved:
                log[key] = value

        return json.dumps(log)



def get_logger(name: str = "inference"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # IMPORTANT: disable propagation to avoid uvicorn swallowing logs
    logger.propagate = False

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    handler.setFormatter(JsonFormatter())

    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()
    logger.addHandler(handler)

    return logger
