from sqlalchemy.orm import Session
from ScoreLens_FastApi.app.model.kafka_model import KafkaMessage, Ball, Collision
from ScoreLens_FastApi.app.request.kafka_request import LogMessageRequest, EventRequest
from typing import List
from ScoreLens_FastApi.app.response.kafka_message_response import KafkaMessageResponse, BallResponse, CollisionResponse
from uuid import uuid4
from typing import Optional


def create_kafka_message(db: Session, message_request: LogMessageRequest):
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

    for collision_req in message_request.collisions:
        collision = Collision(
            ball1=collision_req.ball1,
            ball2=collision_req.ball2,
            time=collision_req.time,
            kafka_message_id=kafka_message.id
        )
        db.add(collision)

    db.commit()
    db.refresh(kafka_message)
    return kafka_message

def get_kafka_messages(db: Session, skip: int = 0, limit: int = 10):
    return db.query(KafkaMessage).offset(skip).limit(limit).all()

def get_kafka_message_by_id(db: Session, message_id: int):
    return db.query(KafkaMessage).filter(KafkaMessage.id == message_id).first()

def delete_kafka_message(db: Session, message_id: int):
    message = get_kafka_message_by_id(db, message_id)
    if message:
        db.delete(message)
        db.commit()
    return message

def delete_kafka_message_by_game_set(db: Session, game_set_id: int):
    msg_list = db.query(KafkaMessage).filter(KafkaMessage.game_set_id == game_set_id).all()
    if not msg_list: return 0
    for tmp in msg_list:
        db.delete(tmp)
    db.commit()
    return len(msg_list)

def delete_kafka_message_by_player(db: Session, player_id: int):
    msg_list = db.query(KafkaMessage).filter(KafkaMessage.player_id == player_id).all()
    if not msg_list: return 0
    for tmp in msg_list:
        db.delete(tmp)
    db.commit()
    return len(msg_list)


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

def convert_kafka_message_to_event_request(message: KafkaMessage) -> EventRequest:
    return EventRequest(
        playerID=message.player_id,
        gameSetID=message.game_set_id,
        scoreValue=message.score_value,
        isFoul=message.is_foul,
        isUncertain=message.is_uncertain,
        message=message.message,
        sceneUrl=message.scene_url
    )


