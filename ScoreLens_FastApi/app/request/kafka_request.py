from pydantic import BaseModel
from typing import List


class Ball(BaseModel):
    id: int
    start: List[int]
    end: List[int]
    potted: bool


class Collision(BaseModel):
    ball1: int
    ball2: int
    time: float


class KafkaMessageRequest(BaseModel):
    timestamp: str
    cueBallId: int
    balls: List[Ball]
    collisions: List[Collision]
    scoreValue: int
    isFoul: bool
    isUncertain: bool
    message: str
    sceneUrl: str
    matchId: int

