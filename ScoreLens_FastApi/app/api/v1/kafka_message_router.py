from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from pydantic import ValidationError
from sqlalchemy.orm import Session
from typing import List

from typing import Annotated

from ScoreLens_FastApi.app.config.deps import get_db  # hàm get_db để lấy session
from ScoreLens_FastApi.app.config.kafka_producer_config import send_json_message, send_json_logging
from ScoreLens_FastApi.app.service import message_service
from ScoreLens_FastApi.app.request.kafka_request import LogMessageRequest, LogMessageCreateRequest, EventRequest
from ScoreLens_FastApi.app.response.kafka_message_response import KafkaMessageResponse  # nếu cần response schema
from ScoreLens_FastApi.app.service.message_service import convert_kafka_message_to_response, \
    convert_kafka_messages_to_responses
import logging
import json

from ScoreLens_FastApi.app.service.s3_service import upload_file_to_s3_with_prefix

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/kafka-messages",
    tags=["Kafka Messages"]
)

# Tạo message mới
@router.post("/", response_model=KafkaMessageResponse, status_code=status.HTTP_201_CREATED)
def create_message(
        log_request: str = Form(
            ...,
            description="Example JSON format for log_request is available in the API doc: api-doc.md",
        ),
        file: UploadFile = File(...),
        db: Session = Depends(get_db)
):
    # parse JSON string sang Pydantic model
    try:
        log_data = json.loads(log_request)
        log_message = LogMessageCreateRequest.model_validate(log_data)
    except (json.JSONDecodeError, ValidationError) as e:
        logger.exception(f"Failed to parse log_request: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid log_request data: {e}")

    # convert from create to msg, để trống scene_url
    try:
        message_request = log_message.to_message_request(scene_url= "")
    except Exception as e:
        logger.exception(f"Failed to convert request: {e}")
        raise HTTPException(status_code=400, detail="Invalid log message data")

    # upload file lên s3
    try:
        file_url = upload_file_to_s3_with_prefix(file.file, "shot", file.filename)
        logger.info(f"Uploaded file to S3 successfully: {file_url}")
    except Exception as e:
        logger.exception(f"Failed to upload file: {e}")
        raise HTTPException(status_code=500, detail="Failed to upload file to S3")

    #add url to req
    message_request.details.sceneUrl = file_url

    # save into db
    try:
        message = message_service.create_kafka_message(db, message_request)
        logger.info("Saved message into DB successfully")
    except Exception as e:
        logger.exception(f"Failed to save message to DB: {e}")
        raise HTTPException(status_code=500, detail="Failed to save message to database")
    # kafka send msg
    try:
        send_json_logging(message_request)
        logger.info("Sent Kafka message successfully")
    except Exception as e:
        logger.exception(f"Failed to send Kafka message: {e}")
        raise HTTPException(status_code=500, detail="Failed to send message to Kafka")

    return convert_kafka_message_to_response(message)


# Lấy danh sách messages
@router.get("/", response_model=List[KafkaMessageResponse])
def list_messages(skip: int = 0, limit: int = 10, db: Session = Depends(get_db)):
    messages = message_service.get_kafka_messages(db, skip, limit)
    return convert_kafka_messages_to_responses(messages)

# Lấy message theo id
@router.get("/{id}", response_model=KafkaMessageResponse)
def get_message_by_id(message_id: int, db: Session = Depends(get_db)):
    message = message_service.get_kafka_message_by_id(db, message_id)
    if not message:
        raise HTTPException(status_code=404, detail="Kafka message not found")
    return convert_kafka_message_to_response(message)

# Xóa message theo id
@router.delete("/{id}", response_model=KafkaMessageResponse)
def delete_message(message_id: int, db: Session = Depends(get_db)):
    message = message_service.delete_kafka_message(db, message_id)
    if not message:
        raise HTTPException(status_code=404, detail="Kafka message not found")
    return convert_kafka_message_to_response(message)

# Xóa theo round
@router.delete("/by-game-set/{game_set_id}")
def delete_message_by_round(game_set_id: int, db: Session = Depends(get_db)):
    msg_list = message_service.delete_kafka_message_by_game_set(db, game_set_id)
    if msg_list == 0:
        raise HTTPException(status_code=404, detail="No messages found to delete for this game set.")
    return {"deleted_count": msg_list}

# Xóa theo player
@router.delete("/by-player/{player_id}")
def delete_message_by_player(player_id: int, db: Session = Depends(get_db)):
    msg_list = message_service.delete_kafka_message_by_player(db, player_id)
    if msg_list == 0:
        raise HTTPException(status_code=404, detail="No messages found to delete for this player.")
    return {"deleted_count": msg_list}























