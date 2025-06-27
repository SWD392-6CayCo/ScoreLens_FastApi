from fastapi import HTTPException
from pydantic import ValidationError, AnyUrl
from sqlalchemy.orm import Session
from ScoreLens_FastApi.app.model.message import KafkaMessage, Ball, Collision
from ScoreLens_FastApi.app.request.message_request import LogMessageRequest, EventRequest, LogMessageCreateRequest, \
    ProducerRequest
from typing import List, Any
from ScoreLens_FastApi.app.response.message_response import KafkaMessageResponse, BallResponse, CollisionResponse
from ScoreLens_FastApi.app.exception.app_exception import AppException
from ScoreLens_FastApi.app.exception.app_exception import ErrorCode
from ScoreLens_FastApi.app.service.s3_service import extract_s3_key_from_url, delete_file_from_s3
import json
import logging

from uuid import uuid4
from typing import Optional

logger = logging.getLogger(__name__)


def create_kafka_message(db: Session, message_request: LogMessageRequest):
    try:
        kafka_message = KafkaMessage(
            cue_ball_id=message_request.cueBallId,
            score_value=message_request.details.scoreValue if message_request.details else False,
            is_foul=message_request.details.isFoul if message_request.details else False,
            is_uncertain=message_request.details.isUncertain if message_request.details else False,
            message=message_request.message,
            scene_url=message_request.details.sceneUrl if message_request.details else "",
            player_id=message_request.details.playerID if message_request.details else 0,
            game_set_id=message_request.details.gameSetID if message_request.details else 0,
        )
        db.add(kafka_message)
        db.flush()  # để lấy id sau khi insert
    except Exception as e:
        logger.exception(f"Failed to create KafkaMessage: {e}")
        raise AppException(
            status_code=500,
            code=ErrorCode.DATABASE_INSERT_ERROR,
            message=f"Failed to create KafkaMessage: {e}"
        )

    try:
        for ball_req in message_request.balls:
            start_x = ball_req.start[0] if len(ball_req.start) > 0 else 0
            start_y = ball_req.start[1] if len(ball_req.start) > 1 else 0
            end_x = ball_req.end[0] if len(ball_req.end) > 0 else 0
            end_y = ball_req.end[1] if len(ball_req.end) > 1 else 0

            ball = Ball(
                start_x=start_x,
                start_y=start_y,
                end_x=end_x,
                end_y=end_y,
                potted=ball_req.potted,
                kafka_message_id=kafka_message.id
            )
            db.add(ball)
    except Exception as e:
        logger.exception(f"Failed to create Ball records: {e}")
        raise AppException(
            status_code=500,
            code=ErrorCode.DATABASE_INSERT_ERROR,
            message=f"Failed to create Ball records: {e}"
        )

    try:
        for collision_req in message_request.collisions:
            collision = Collision(
                ball1=collision_req.ball1,
                ball2=collision_req.ball2,
                time=collision_req.time,
                kafka_message_id=kafka_message.id
            )
            db.add(collision)
    except Exception as e:
        logger.exception(f"Failed to create Collision records: {e}")
        raise AppException(
            status_code=500,
            code=ErrorCode.DATABASE_INSERT_ERROR,
            message=f"Failed to create Collision records: {e}"
        )

    try:
        db.commit()
        db.refresh(kafka_message)
        logger.info(f"Kafka message created successfully with ID {kafka_message.id}")
        return kafka_message
    except Exception as e:
        logger.exception(f"Failed to commit transaction: {e}")
        db.rollback()
        raise AppException(
            status_code=500,
            code=ErrorCode.DATABASE_COMMIT_ERROR,
            message=f"Failed to commit Kafka message transaction: {e}"
        )

def get_kafka_messages(db: Session, skip: int = 0, limit: int = 10):
    try:
        messages = db.query(KafkaMessage).offset(skip).limit(limit).all()
        if not messages:
            raise AppException(
                status_code=404,
                code=ErrorCode.MESSAGE_LIST_NOT_FOUND,
                message=f"Kafka message list is null."
            )
        return messages
    except AppException as ae:
        raise ae  # giữ nguyên exception custom của mình
    except Exception as e:
        logger.exception(f"Failed to fetch Kafka messages by ID: {e}")
        raise AppException(
            status_code=500,
            code=ErrorCode.DATABASE_QUERY_ERROR,
            message=f"Failed to fetch Kafka messages by ID: {e}"
        )

def get_kafka_message_by_id(db: Session, message_id: int):
    try:
        message = db.query(KafkaMessage).filter(KafkaMessage.id == message_id).first()
        if not message:
            raise AppException(
                status_code=404,
                code=ErrorCode.MESSAGE_NOT_FOUND,
                message=f"Kafka message with id {message_id} not found."
            )
        return message
    except Exception as e:
        logger.exception(f"Failed to fetch Kafka message by ID: {e}")
        raise AppException(
            status_code=500,
            code=ErrorCode.DATABASE_QUERY_ERROR,
            message=f"Failed to fetch Kafka message by ID: {e}"
        )

def get_kafka_messages_by_player_id(db: Session, player_id: int):
    try:
        msg_list = db.query(KafkaMessage).filter(KafkaMessage.player_id == player_id).all()
        if not msg_list:
            raise AppException(
                status_code=404,
                code=ErrorCode.MESSAGE_NOT_FOUND,
                message=f"Kafka message list with player_id {player_id} not found."
            )
        return msg_list
    except AppException as ae:
        logger.exception(f"Failed to fetch Kafka message by playerID: {ae}")
        raise AppException(
            status_code=500,
            code=ErrorCode.DATABASE_QUERY_ERROR,
            message=f"Failed to fetch Kafka message by playerID: {ae}"
        )

def get_kafka_messages_by_game_set_id(db: Session, game_set_id: int):
    try:
        msg_list = db.query(KafkaMessage).filter(KafkaMessage.game_set_id == game_set_id).all()
        if not msg_list:
            raise AppException(
                status_code=404,
                code=ErrorCode.MESSAGE_NOT_FOUND,
                message=f"Kafka message list with game_set_id {game_set_id} not found."
            )
        return msg_list
    except AppException as ae:
        logger.exception(f"Failed to fetch Kafka message by playerID: {ae}")
        raise AppException(
            status_code=500,
            code=ErrorCode.DATABASE_QUERY_ERROR,
            message=f"Failed to fetch Kafka message by playerID: {ae}"
        )

def  delete_kafka_message(db: Session, message_id: int):
    try:
        message = get_kafka_message_by_id(db, message_id)
        tmp: Any = message.scene_url
        if tmp:
            delete_scene_url(tmp)
        db.delete(message)
        db.commit()
        logger.info(f"Deleted Kafka message with ID {message_id}")
        return "Deleted Kafka message with ID {message_id}", message_id
    except Exception as e:
        logger.exception(f"Failed to delete Kafka message with ID {message_id}: {e}")
        raise AppException(
            status_code=500,
            code=ErrorCode.DATABASE_DELETE_ERROR,
            message=f"Failed to delete Kafka message with ID {message_id}: {e}"
        )


def delete_kafka_message_by_game_set(db: Session, game_set_id: int):
    try:
        msg_list = get_kafka_messages_by_game_set_id(db, game_set_id)
        for msg in msg_list:
            tmp: Any = msg.scene_url
            if tmp:
                delete_scene_url(tmp)
            db.delete(msg)
        db.commit()
        logger.info(f"Deleted {len(msg_list)} Kafka messages for game_set_id {game_set_id}.")
        return len(msg_list)
    except Exception as e:
        logger.exception(f"Failed to delete Kafka messages for game_set_id {game_set_id}: {e}")
        raise AppException(
            status_code=500,
            code=ErrorCode.DATABASE_DELETE_ERROR,
            message=f"Failed to delete Kafka messages for game_set_id {game_set_id}: {e}"
        )


def delete_kafka_message_by_player(db: Session, player_id: int):
    try:
        msg_list = get_kafka_messages_by_player_id(db, player_id)
        for msg in msg_list:
            tmp: Any = msg.scene_url
            if tmp:
                delete_scene_url(tmp)
            db.delete(msg)
        db.commit()
        logger.info(f"Deleted {len(msg_list)} Kafka messages for player_id {player_id}.")
        return len(msg_list)
    except Exception as e:
        logger.exception(f"Failed to delete Kafka messages for player_id {player_id}: {e}")
        raise AppException(
            status_code=500,
            code=ErrorCode.DATABASE_DELETE_ERROR,
            message=f"Failed to delete Kafka messages for player_id {player_id}: {e}"
        )


def parse_json_to_producer_request(json_string: str) -> ProducerRequest:
    try:
        return ProducerRequest.model_validate_json(json_string)
    except ValidationError as e:
        logger.exception("Validation error while parsing log_request")
        raise AppException(
            status_code=400,
            code=ErrorCode.VALIDATION_ERROR,
            message=f"Invalid log_request schema: {e}"
        )
    except Exception as e:
        logger.exception("Unexpected error while parsing log_request")
        raise AppException(
            status_code=500,
            code=ErrorCode.UNKNOWN_ERROR,
            message=f"Unexpected error while parsing log_request: {str(e)}"
        )



def convert_producer_request_to_log_message_create_request(producer_request: ProducerRequest) -> LogMessageCreateRequest:
    """
    Chuyển đổi dữ liệu từ ProducerRequest thành LogMessageCreateRequest.
    """
    if not isinstance(producer_request.data, dict):
        raise AppException(
            status_code=400,
            code=ErrorCode.VALIDATION_ERROR,
            message="ProducerRequest.data must be a JSON object (dict)"
        )

    try:
        log_message_req = LogMessageCreateRequest.model_validate(producer_request.data)
        return log_message_req

    except ValidationError as e:
        logger.exception("Validation error when converting data to LogMessageCreateRequest")
        raise AppException(
            status_code=400,
            code=ErrorCode.VALIDATION_ERROR,
            message=f"Invalid log_request data schema: {e}"
        )

    


# convert from create to msg
def convert_create_to_msg(req: LogMessageCreateRequest) -> LogMessageRequest:
    try:
        message_request = req.to_message_request(scene_url="")
        return message_request
    except AttributeError as e:
        logger.exception(f"Attribute error while converting request: {e}")
        raise AppException(
            status_code=400,
            code=ErrorCode.VALIDATION_ERROR,
            message=f"Invalid log message data structure: {e}"
        )
    except Exception as e:
        logger.exception(f"Unexpected error while converting request: {e}")
        raise AppException(
            status_code=500,
            code=ErrorCode.UNKNOWN_ERROR,
            message=f"Unexpected error while converting request: {e}"
        )

#delete image on s3
def delete_scene_url(scene_url: str) -> None:
    string = extract_s3_key_from_url(scene_url)
    delete_file_from_s3(string)


#***************************************** mapper ***********************************************
def convert_kafka_message_to_response(kafka_message: KafkaMessage) -> KafkaMessageResponse:
    return KafkaMessageResponse(
        id=kafka_message.id,
        timestamp=kafka_message.timestamp,
        cueBallId=kafka_message.cue_ball_id,
        balls=[
            BallResponse(
                start=[ball.start_x, ball.start_y],
                end=[ball.end_x, ball.end_y],
                potted=ball.potted
            )
            for ball in kafka_message.balls
        ],
        collisions=[
            CollisionResponse(
                ball1=col.ball1,
                ball2=col.ball2,
                time=col.time
            )
            for col in kafka_message.collisions
        ],
        scoreValue=kafka_message.score_value,
        isFoul=kafka_message.is_foul,
        isUncertain=kafka_message.is_uncertain,
        message=kafka_message.message,
        sceneUrl=kafka_message.scene_url,
        playerId=kafka_message.player_id,
        gameSetId=kafka_message.game_set_id
    )

def convert_kafka_messages_to_responses(kafka_messages: List[KafkaMessage]) -> List[KafkaMessageResponse]:
    return [convert_kafka_message_to_response(message) for message in kafka_messages]



