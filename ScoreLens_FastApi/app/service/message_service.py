from sqlalchemy import Column
from sqlalchemy.orm import Session
from ScoreLens_FastApi.app.model.kafka_model import KafkaMessage, Ball, Collision
from ScoreLens_FastApi.app.request.kafka_request import KafkaMessageRequest
from sqlalchemy import DateTime
from sqlalchemy.sql import func

def create_kafka_message(db: Session, message_request: KafkaMessageRequest):
    kafka_message = KafkaMessage(
        timestamp=Column(DateTime(timezone=True), server_default=func.now()),
        cue_ball_id=message_request.cueBallId,
        score_value=message_request.scoreValue,
        is_foul=message_request.isFoul,
        is_uncertain=message_request.isUncertain,
        message=message_request.message,
        scene_url=message_request.sceneUrl,
        match_id=message_request.matchId
    )
    db.add(kafka_message)
    db.flush()  # để lấy id sau khi insert

    for ball_req in message_request.balls:
        ball = Ball(
            start_x=ball_req.start[0],
            start_y=ball_req.start[1],
            end_x=ball_req.end[0],
            end_y=ball_req.end[1],
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
