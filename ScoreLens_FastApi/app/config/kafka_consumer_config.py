import os
import json
from functools import lru_cache
from dotenv import load_dotenv
from kafka import KafkaConsumer
load_dotenv()

TOPIC_CONSUMER = os.getenv("KAFKA_TOPIC_CONSUMER")
BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS")
SSL_CA_CERT = os.getenv("SSL_CA_CERT")
SSL_CERTFILE = os.getenv("SSL_CERTFILE")
SSL_KEYFILE = os.getenv("SSL_KEYFILE")
GROUP_ID = 'fastapi-consumer-group'
TABLE_ID_KEY = os.getenv("KAFKA_TABLE_ID_KEY")

# Khởi tạo consumer 1 lần duy nhất
@lru_cache
def consumer():
    return KafkaConsumer(
        TOPIC_CONSUMER,
        bootstrap_servers=BOOTSTRAP_SERVERS,
        security_protocol="SSL",
        ssl_cafile=SSL_CA_CERT,
        ssl_certfile=SSL_CERTFILE,
        ssl_keyfile=SSL_KEYFILE,
        auto_offset_reset='earliest',
        enable_auto_commit=True,
        # group_id=GROUP_ID,
        value_deserializer=lambda m: json.loads(m.decode('utf-8')),
        connections_max_idle_ms=600000
    )


