import logging
import json
import asyncio  # Import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import ValidationError

from ScoreLens_FastApi.app.api.v1 import s3_router, message_router
from ScoreLens_FastApi.app.exception.app_exception import AppException
from ScoreLens_FastApi.app.exception.global_exception_handler import app_exception_handler, \
    validation_exception_handler, json_exception_handler
from ScoreLens_FastApi.app.service.kafka_consumer_service import consume_all_partitions
from ScoreLens_FastApi.app.state_manager_class.detect_state import stop_all_detection_tasks

############################################# Cấu hình logging #######################################################
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Biến toàn cục để giữ tác vụ consumer
consumer_task = None


############################################# background task consumer ###############################################
# thay thế cho on_event() handling
@asynccontextmanager
async def lifespan(app: FastAPI):
    global consumer_task
    # ---- Startup phase ----
    # SỬA ĐỔI: Chạy consume_all_partitions như một asyncio task
    consumer_task = asyncio.create_task(consume_all_partitions())
    logger.info("Kafka consumer background task started.")

    yield  # <-- pause tại đây, app FastAPI bắt đầu nhận request

    # ---- Shutdown phase ----
    logger.info("FastAPI app shutting down.")
    # Hủy tác vụ consumer khi ứng dụng tắt
    if consumer_task:
        consumer_task.cancel()
        try:
            # SỬA ĐỔI: Thêm timeout cho việc đợi tác vụ consumer dừng
            await asyncio.wait_for(consumer_task, timeout=10.0)  # Đợi tối đa 10 giây
            logger.info("Kafka consumer background task stopped.")
        except asyncio.CancelledError:
            logger.info("Kafka consumer background task confirmed cancelled.")
        except asyncio.TimeoutError:
            logger.warning("Kafka consumer background task timed out during shutdown. It might still be running.")
        except Exception as e:
            logger.error(f"Error stopping Kafka consumer task: {e}")

    # Đảm bảo tất cả các tác vụ phát hiện cũng dừng
    await stop_all_detection_tasks()
    logger.info("All YOLOv8 detection tasks stopped during shutdown.")


#################################################### FastApi ########################################################
app = FastAPI(title="FastApi", version="1.0.0", lifespan=lifespan)

# Exception Handlers
app.add_exception_handler(AppException, app_exception_handler)
app.add_exception_handler(ValidationError, validation_exception_handler)
app.add_exception_handler(json.JSONDecodeError, json_exception_handler)


@app.get("/health")
def health_check():
    return {"status": "ok"}


# Include routers
app.include_router(s3_router.router)
app.include_router(message_router.router)

