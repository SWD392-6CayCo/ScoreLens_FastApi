from ScoreLens_FastApi.app.exception.app_exception import AppException
from ScoreLens_FastApi.app.exception.error_code import ErrorCode
from ScoreLens_FastApi.app.request.kafka_request import LogMessageRequest, ProducerRequest
from ScoreLens_FastApi.app.config.kafka_producer_config import producer, TOPIC_PRODUCER

import logging

logger = logging.getLogger(__name__)


def send_to_java(msg: ProducerRequest):
    try:
        p = producer()
        p.send(TOPIC_PRODUCER, value=msg.model_dump_json(), partition=0)
        p.flush()
        logger.info(f"Sent Kafka message to topic {TOPIC_PRODUCER}: {msg}")
    except Exception as e:
        logger.exception(f"Failed to send message to Kafka: {e}")
        raise AppException(
            status_code=500,
            code=ErrorCode.KAFKA_SEND_FAILED,
            message=str(e)
        )