from sqlalchemy import Column, Integer, String, Boolean, Float, ForeignKey
from sqlalchemy.orm import relationship
from ScoreLens_FastApi.app.config.db import Base

class KafkaMessage(Base):
    __tablename__ = "kafka_message"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(String)
    cue_ball_id = Column(Integer)
    score_value = Column(Integer)
    is_foul = Column(Boolean)
    is_uncertain = Column(Boolean)
    message = Column(String)
    scene_url = Column(String)
    match_id = Column(Integer)

    balls = relationship("Ball", back_populates="kafka_message")
    collisions = relationship("Collision", back_populates="kafka_message")


class Ball(Base):
    __tablename__ = "ball"

    id = Column(Integer, primary_key=True, index=True)
    start_x = Column(Integer)
    start_y = Column(Integer)
    end_x = Column(Integer)
    end_y = Column(Integer)
    potted = Column(Boolean)
    kafka_message_id = Column(Integer, ForeignKey("kafka_message.id"))

    kafka_message = relationship("KafkaMessage", back_populates="balls")


class Collision(Base):
    __tablename__ = "collision"

    id = Column(Integer, primary_key=True, index=True)
    ball1 = Column(Integer)
    ball2 = Column(Integer)
    time = Column(Float)
    kafka_message_id = Column(Integer, ForeignKey("kafka_message.id"))

    kafka_message = relationship("KafkaMessage", back_populates="collisions")
