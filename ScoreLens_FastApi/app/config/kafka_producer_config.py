import os
from functools import lru_cache

from dotenv import load_dotenv
from kafka import KafkaProducer


from ScoreLens_FastApi.app.request.kafka_request import EventRequest, LogMessageRequest

load_dotenv()

TOPIC_PRODUCER = os.getenv("KAFKA_TOPIC_PRODUCER")

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


# đảm bảo các message còn nằm trong buffer của producer được gửi hết về Kafka broker ngay lập tức.
def flush_producer():
    producer.flush()

# đóng kết nối producer
def close_producer():
    producer.close()






