# app/config.py

from pydantic_settings import BaseSettings

class KafkaSettings(BaseSettings):
    BOOTSTRAP_SERVERS: str = 'kafka-5c346d1-kafka-scorelens.f.aivencloud.com:26036'
    CA_CERT: str = 'certs/ca.pem'
    ACCESS_CERT: str = 'certs/access.pem'
    ACCESS_KEY: str = 'certs/access.key'
    TOPIC_NAME: str = 'scorelens'

    class Config:
        env_file = ".env"

kafka_settings = KafkaSettings()

