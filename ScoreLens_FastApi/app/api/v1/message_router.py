from fastapi import APIRouter, Depends, status, UploadFile, File, Form
from sqlalchemy.orm import Session
from typing import List
from ScoreLens_FastApi.app.config.deps import get_db  # hàm get_db để lấy session
from ScoreLens_FastApi.app.request.message_request import ProducerRequest
from ScoreLens_FastApi.app.service import message_service
from ScoreLens_FastApi.app.response.message_response import KafkaMessageResponse  # nếu cần response schema
from ScoreLens_FastApi.app.service.kafka_producer_service import send_to_java
from ScoreLens_FastApi.app.service.message_service import convert_kafka_message_to_response, \
    convert_kafka_messages_to_responses, convert_create_to_msg, parse_json_to_producer_request, \
    convert_producer_request_to_log_message_create_request
from ScoreLens_FastApi.app.service.s3_service import upload_file_to_s3_with_prefix
import logging


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
    # parse json string to Pydantic ProducerRequest and parse PR data to LogMsgReq
    tmp = parse_json_to_producer_request(log_request)
    log_message = convert_producer_request_to_log_message_create_request(tmp)

    # convert from create to msg, để trống scene_url
    message_request = convert_create_to_msg(log_message)

    # upload file lên s3
    file_url = upload_file_to_s3_with_prefix(file.file, "shot", file.filename)

    #add url to req
    message_request.details.sceneUrl = file_url

    # save into db
    message = message_service.create_kafka_message(db, message_request)

    #add url to tmp
    tmp.data = message_request

    # kafka send msg
    send_to_java(tmp)

    return convert_kafka_message_to_response(message)



# Lấy danh sách messages
@router.get("/", response_model=List[KafkaMessageResponse])
def list_messages(skip: int = 0, limit: int = 10, db: Session = Depends(get_db)):
    list = message_service.get_kafka_messages(db, skip, limit)
    return convert_kafka_messages_to_responses(list)

# Lấy message theo id
@router.get("/{id}", response_model=KafkaMessageResponse)
def list_message_by_id(message_id: int, db: Session = Depends(get_db)):
    list = message_service.get_kafka_message_by_id(db, message_id)
    return convert_kafka_message_to_response(list)

# Lấy messages theo player_id
@router.get("/player/{player_id}", response_model=List[KafkaMessageResponse])
def list_message_by_player_id(player_id: int, db: Session = Depends(get_db)):
    list = message_service.get_kafka_messages_by_player_id(db, player_id)
    return convert_kafka_messages_to_responses(list)

# Lấy messages theo game_set_id
@router.get("/game_set/{game_set_id}", response_model=List[KafkaMessageResponse])
def list_message_by_player_id(game_set_id: int, db: Session = Depends(get_db)):
    list = message_service.get_kafka_messages_by_game_set_id(db, game_set_id)
    return convert_kafka_messages_to_responses(list)


# Xóa message theo id
@router.delete("/{id}")
def delete_message(message_id: int, db: Session = Depends(get_db)):
    message = message_service.delete_kafka_message(db, message_id)
    return message

# Xóa theo round
@router.delete("/by-game-set/{game_set_id}")
def delete_message_by_round(game_set_id: int, db: Session = Depends(get_db)):
    msg_list = message_service.delete_kafka_message_by_game_set(db, game_set_id)
    return {"deleted_count": msg_list}

# Xóa theo player
@router.delete("/by-player/{player_id}")
def delete_message_by_player(player_id: int, db: Session = Depends(get_db)):
    msg_list = message_service.delete_kafka_message_by_player(db, player_id)
    return {"deleted_count": msg_list}























