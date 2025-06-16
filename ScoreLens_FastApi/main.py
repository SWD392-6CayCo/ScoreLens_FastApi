import time

from fastapi import FastAPI
from kafka import KafkaProducer

from kafka_producer import send_json_message
from kafka_request import KafkaMessageRequest

app =FastAPI()

# json api
@app.post("/scorelens/json")
def scorelens(request: KafkaMessageRequest):
    send_json_message(request)
    return {"message": request}
