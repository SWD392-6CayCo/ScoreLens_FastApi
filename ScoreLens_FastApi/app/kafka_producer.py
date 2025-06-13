# app/kafka_producer.py
#
from app.config import kafka_settings
from kafka import KafkaProducer

TOPIC_NAME = "scorelens"

def create_producer():
    return KafkaProducer(
        bootstrap_servers=kafka_settings.BOOTSTRAP_SERVERS,
        security_protocol="SSL",
        ssl_cafile=kafka_settings.CA_CERT,
        ssl_certfile=kafka_settings.ACCESS_CERT,
        ssl_keyfile=kafka_settings.ACCESS_KEY,
    )


def send_message(topic: str, message: str):
    producer = create_producer()
    producer.send(topic, message.encode('utf-8'))
    producer.flush()
    producer.close()