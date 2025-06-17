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

class KafkaMessageResponse(BaseModel):
    id: int
    timestamp: datetime
    cueBallId: int
    balls: List[BallResponse]
    collisions: List[CollisionResponse]
    scoreValue: int
    isFoul: bool
    isUncertain: bool
    message: str
    sceneUrl: str
    matchId: int

    class Config:
        from_attributes = True
