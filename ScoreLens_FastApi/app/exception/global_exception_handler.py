from fastapi import Request
from fastapi.responses import JSONResponse, Response
from pydantic import ValidationError
import json
import logging

from .app_exception import AppException

logger = logging.getLogger(__name__)

def app_exception_handler(request: Request, exc: AppException) -> Response:
    logger.error(f"AppException: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.detail
    )

def validation_exception_handler(request: Request, exc: ValidationError) -> Response:
    logger.error(f"Validation Error: {exc.errors()}")
    return JSONResponse(
        status_code=400,
        content={"code": "VALIDATION_ERROR", "message": str(exc)}
    )

def json_exception_handler(request: Request, exc: json.JSONDecodeError) -> Response:
    logger.error(f"JSON Decode Error: {exc}")
    return JSONResponse(
        status_code=400,
        content={"code": "JSON_DECODE_ERROR", "message": str(exc)}
    )
