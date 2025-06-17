from pydantic import BaseModel
from typing import List


class Ball(BaseModel):
    start: List[int]
    end: List[int]
    potted: bool


class Collision(BaseModel):
    ball1: int
    ball2: int
    time: float


class KafkaMessageRequest(BaseModel):
    cueBallId: int
    balls: List[Ball]
    collisions: List[Collision]
    scoreValue: bool
    isFoul: bool
    isUncertain: bool
    message: str
    sceneUrl: str
    playerId: int
    roundId: int

class EventRequest(BaseModel):
    playerId: int
    roundId: int
    scoreValue: bool
    isFoul: bool
    isUncertain: bool
    message: str
    sceneUrl: str

