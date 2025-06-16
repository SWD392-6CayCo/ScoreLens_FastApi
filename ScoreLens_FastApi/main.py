import time

from fastapi import FastAPI
from kafka import KafkaProducer

from kafka_request import KafkaMessageRequest
from kafka_producer import send_json_message


data = {
    "timestamp": "2025-06-16T21:45:00Z",
    "cueBallId": 0,
    "balls": [
        {"id": 1, "start": [100, 200], "end": [400, 500], "potted": False},
        {"id": 2, "start": [150, 300], "end": [150, 300], "potted": True}
    ],
    "collisions": [
        {"ball1": 0, "ball2": 1, "time": 0.2},
        {"ball1": 0, "ball2": 2, "time": 0.5}
    ],
    "scoreValue": 1,
    "isFoul": False,
    "isUncertain": False,
    "message": "Ball 2 potted after collision.",
    "sceneUrl": "https://scorelens/frames/shot123.jpg",
    "matchId": 101
}

# **************************** FastAPI ******************************
app =FastAPI()

# json api
@app.post("/scorelens/json")
def scorelens(request: KafkaMessageRequest):
    send_json_message(request)
    return {"message": request}
