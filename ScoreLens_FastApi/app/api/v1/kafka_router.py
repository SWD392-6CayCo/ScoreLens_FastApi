from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import logging

from ScoreLens_FastApi.app.config.kafka_producer_config import send_json_message
from ScoreLens_FastApi.app.request.kafka_request import EventRequest

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/scorelens", tags=["Kafka"])

@router.post("/scorelens/json", response_model=Dict[str, Any])
def scorelens(request: EventRequest):
    """
    Gửi JSON message vào Kafka topic.
    """
    try:
        send_json_message(request)
        logger.info("Sent Kafka message successfully")
        return {"message": request}
    except Exception as e:
        logger.error(f"Failed to send Kafka message: {e}")
        raise HTTPException(status_code=500, detail="Failed to send Kafka message")
