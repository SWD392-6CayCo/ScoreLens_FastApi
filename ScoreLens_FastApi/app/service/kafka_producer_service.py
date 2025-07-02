from ScoreLens_FastApi.app.exception.app_exception import AppException
from ScoreLens_FastApi.app.exception.error_code import ErrorCode
from ScoreLens_FastApi.app.request.message_request import LogMessageRequest, ProducerRequest
from ScoreLens_FastApi.app.config.kafka_producer_config import producer, TOPIC_PRODUCER

import logging

logger = logging.getLogger(__name__)


def send_to_java(msg: ProducerRequest, table_id: str):
    try:
        p = producer()

        key_bytes = table_id.encode('utf-8') if table_id else None

        p.send(
            TOPIC_PRODUCER,
            key=key_bytes,
            value=msg.model_dump_json()
        )
        p.flush()
        logger.info(f"Sent Kafka message to topic {TOPIC_PRODUCER} with key {table_id}: {msg}")
    except Exception as e:
        logger.exception(f"Failed to send message to Kafka: {e}")
        raise AppException(
            status_code=500,
            code=ErrorCode.KAFKA_SEND_FAILED,
            message=str(e)
        )