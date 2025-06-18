from datetime import datetime
from pydantic import BaseModel
from typing import List

class BallResponse(BaseModel):
    start: List[int]
    end: List[int]
    potted: bool

class CollisionResponse(BaseModel):
    ball1: int
    ball2: int
    time: float

class BaseEventData(BaseModel):
    playerId: int
    roundId: int
    scoreValue: bool
    isFoul: bool
    isUncertain: bool
    message: str
    sceneUrl: str

class KafkaMessageResponse(BaseModel):
    id: int
    timestamp: datetime
    cueBallId: int
    balls: List[BallResponse]
    collisions: List[CollisionResponse]
    scoreValue: bool
    isFoul: bool
    isUncertain: bool
    message: str
    sceneUrl: str
    playerId: int
    roundId: int

    class Config:
        from_attributes = True
