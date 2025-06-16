import json
import os
from functools import lru_cache

from dotenv import load_dotenv
from kafka import KafkaProducer

from kafka_request import KafkaMessageRequest

load_dotenv()

TOPIC_NAME = os.getenv("KAFKA_TOPIC")

# khỏi tạo producer một lần duy nhất
@lru_cache
def producer():
    return KafkaProducer(
        bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP_SERVERS"),
        security_protocol="SSL",
        ssl_cafile=os.getenv("SSL_CA_CERT"),
        ssl_certfile=os.getenv("SSL_CERTFILE"),
        ssl_keyfile=os.getenv("SSL_KEYFILE"),
        value_serializer=lambda v: v.encode("utf-8"),
    )

def send_json_message(msg: KafkaMessageRequest):
    p = producer()
    p.send(TOPIC_NAME, value=msg.model_dump_json())
    p.flush()


# đảm bảo các message còn nằm trong buffer của producer được gửi hết về Kafka broker ngay lập tức.
def flush_producer():
    producer.flush()

# đóng kết nối producer
def close_producer():
    producer.close()






