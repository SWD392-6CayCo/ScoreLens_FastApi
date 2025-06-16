import json
from functools import lru_cache

from kafka import KafkaProducer

from kafka_request import KafkaMessageRequest

TOPIC_NAME = "scorelens"

# khỏi tạo producer một lần duy nhất
@lru_cache
def producer():
    return KafkaProducer(
        bootstrap_servers=f"kafka-5c346d1-kafka-scorelens.f.aivencloud.com:26036",
        security_protocol="SSL",
        ssl_cafile="certs/ca.pem",
        ssl_certfile="certs/service.cert",
        ssl_keyfile="certs/service.key",
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )

def send_json_message(msg: KafkaMessageRequest):
    producer().send(TOPIC_NAME, value=msg.model_dump())

# đảm bảo các message còn nằm trong buffer của producer được gửi hết về Kafka broker ngay lập tức.
def flush_producer():
    producer.flush()

# đóng kết nối producer
def close_producer():
    producer.close()






