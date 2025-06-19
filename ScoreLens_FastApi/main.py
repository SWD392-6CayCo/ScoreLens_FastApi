import logging
from contextlib import asynccontextmanager
from threading import Thread

from fastapi import FastAPI
from ScoreLens_FastApi.app.api.v1 import kafka_router, s3_router, kafka_message_router
from ScoreLens_FastApi.app.service.kafka_consumer_service import consume_messages

############################################# Cấu hình logging #######################################################
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

############################################# background task consumer ###############################################
# thay thế cho on_event() handling
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ---- Startup phase ----
    consumer_thread = Thread(target=consume_messages, daemon=True)
    consumer_thread.start()
    logger.info("Kafka consumer background task started.")

    yield  # <-- pause tại đây, app FastAPI bắt đầu nhận request

    # ---- Shutdown phase ----
    logger.info("FastAPI app shutting down.")

#################################################### FastApi ########################################################
app = FastAPI(title="FastApi",version="1.0.0", lifespan=lifespan)
@app.get("/health")
def health_check():
    return {"status": "ok"}

# Include routers
app.include_router(kafka_router.router)
app.include_router(s3_router.router)
app.include_router(kafka_message_router.router)






