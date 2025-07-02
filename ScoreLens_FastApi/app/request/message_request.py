from datetime import datetime
from uuid import uuid4

from pydantic import BaseModel, Field
from typing import List, Optional, Any

from pydantic_core.core_schema import json_schema

from ScoreLens_FastApi.app.enum.kafka_code import KafkaCode


class Ball(BaseModel):
    start: List[int]
    end: List[int]
    potted: bool


class Collision(BaseModel):
    ball1: int
    ball2: int
    time: float

class EventCreateRequest(BaseModel):
    playerID: int
    gameSetID: int
    scoreValue: bool
    isFoul: bool
    isUncertain: bool
    message: str

class EventRequest(EventCreateRequest):
    sceneUrl: str



class LogMessageCreateRequest(BaseModel):
    level: str
    type: str
    cueBallId: int
    balls: List[Ball]
    collisions: List[Collision]
    message: str
    details: Optional[EventCreateRequest] = None

    #convert create req to req
    def to_message_request(self, scene_url: str) -> "LogMessageRequest":
        # convert details nếu có
        event_details = None
        if self.details:
            event_details = EventRequest(
                **self.details.model_dump(),
                sceneUrl=scene_url
            )

        return LogMessageRequest(
            level=self.level,
            type=self.type,
            cueBallId=self.cueBallId,
            balls=self.balls,
            collisions=self.collisions,
            message=self.message,
            details=event_details
        )

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

class ProducerRequest(BaseModel):
    code: KafkaCode
    tableID: str
    data: Any

    class Config:
        use_enum_values = True



