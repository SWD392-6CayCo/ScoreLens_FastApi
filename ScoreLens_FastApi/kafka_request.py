from pydantic import BaseModel


class KafkaMessageRequest(BaseModel):
    player_id: int
    score: bool
    message: str
