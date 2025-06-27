from datetime import datetime

from sqlalchemy import Column, Integer, String, Boolean, Float, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from ScoreLens_FastApi.app.config.db import Base

class KafkaMessage(Base):
    __tablename__ = "message"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.now())
    cue_ball_id = Column(Integer)
    score_value = Column(Boolean)
    is_foul = Column(Boolean)
    is_uncertain = Column(Boolean)
    message = Column(String)
    scene_url = Column(String)
    player_id = Column(Integer)
    game_set_id = Column(Integer)

    balls = relationship("Ball", back_populates="message", cascade="all, delete-orphan")
    collisions = relationship("Collision", back_populates="message", cascade="all, delete-orphan")


class Ball(Base):
    __tablename__ = "ball"

    id = Column(Integer, primary_key=True, index=True)
    start_x = Column(Integer)
    start_y = Column(Integer)
    end_x = Column(Integer)
    end_y = Column(Integer)
    potted = Column(Boolean)
    kafka_message_id = Column(Integer, ForeignKey("message.id"))

    kafka_message = relationship("KafkaMessage", back_populates="balls")


class Collision(Base):
    __tablename__ = "collision"

    id = Column(Integer, primary_key=True, index=True)
    ball1 = Column(Integer)
    ball2 = Column(Integer)
    time = Column(Float)
    kafka_message_id = Column(Integer, ForeignKey("message.id"))

    kafka_message = relationship("KafkaMessage", back_populates="collisions")
