import logging
import json
from contextlib import asynccontextmanager
from threading import Thread

from fastapi import FastAPI
from pydantic import ValidationError

from ScoreLens_FastApi.app.api.v1 import s3_router, kafka_message_router, detect_router
from ScoreLens_FastApi.app.exception.app_exception import AppException
from ScoreLens_FastApi.app.exception.global_exception_handler import app_exception_handler, \
    validation_exception_handler, json_exception_handler
from ScoreLens_FastApi.app.service.kafka_consumer_service import consume_partition

############################################# Cấu hình logging #######################################################
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

############################################# background task consumer ###############################################
# thay thế cho on_event() handling
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ---- Startup phase ----
    consumer_thread = Thread(target=lambda: consume_partition(0), daemon=True)
    consumer_thread.start()
    logger.info("Kafka consumer background task started.")

    yield  # <-- pause tại đây, app FastAPI bắt đầu nhận request

    # ---- Shutdown phase ----
    logger.info("FastAPI app shutting down.")

#################################################### FastApi ########################################################
app = FastAPI(title="FastApi",version="1.0.0", lifespan=lifespan)

# Exception Handlers
app.add_exception_handler(AppException, app_exception_handler)
app.add_exception_handler(ValidationError, validation_exception_handler)
app.add_exception_handler(json.JSONDecodeError, json_exception_handler)

@app.get("/health")
def health_check():
    return {"status": "ok"}

# Include routers
app.include_router(s3_router.router)
app.include_router(kafka_message_router.router)
app.include_router(detect_router.router)









