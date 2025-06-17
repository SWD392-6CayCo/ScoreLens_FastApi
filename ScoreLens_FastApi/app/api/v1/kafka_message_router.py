from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

from ScoreLens_FastApi.app.config.deps import get_db  # hàm get_db để lấy session
from ScoreLens_FastApi.app.service import message_service
from ScoreLens_FastApi.app.request.kafka_request import KafkaMessageRequest
from ScoreLens_FastApi.app.response.kafka_message_response import KafkaMessageResponse  # nếu cần response schema
from ScoreLens_FastApi.app.service.message_service import convert_kafka_message_to_response, \
    convert_kafka_messages_to_responses

router = APIRouter(
    prefix="/kafka-messages",
    tags=["Kafka Messages"]
)

# Tạo message mới
@router.post("/", response_model=KafkaMessageResponse, status_code=status.HTTP_201_CREATED)
def create_message(message_request: KafkaMessageRequest, db: Session = Depends(get_db)):
    message = message_service.create_kafka_message(db, message_request)
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
