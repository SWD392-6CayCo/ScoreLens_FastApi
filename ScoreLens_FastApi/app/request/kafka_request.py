from datetime import datetime
from uuid import uuid4

from pydantic import BaseModel, Field
from typing import List, Optional


class Ball(BaseModel):
    start: List[int]
    end: List[int]
    potted: bool


class Collision(BaseModel):
    ball1: int
    ball2: int
    time: float

class EventRequest(BaseModel):
    playerID: int
    gameSetID: int
    scoreValue: bool
    isFoul: bool
    isUncertain: bool
    message: str
    sceneUrl: str

class LogMessageCreateRequest(BaseModel):
    level: str
    type: str
    cueBallId: int
    balls: List[Ball]
    collisions: List[Collision]
    message: str
    details: Optional[EventRequest] = None

class LogMessageRequest(BaseModel):
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    service: str = "fastapi-ai-camera"
    level: str
    type: str
    trace_id: str = Field(default_factory=lambda: str(uuid4()))
    cueBallId: int
    balls: List[Ball]
    collisions: List[Collision]
    message: str
    details: Optional[EventRequest] = None

