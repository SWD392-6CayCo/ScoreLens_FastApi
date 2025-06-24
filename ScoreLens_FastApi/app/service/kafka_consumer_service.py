from kafka import TopicPartition
from ScoreLens_FastApi.app.config.kafka_consumer_config import consumer, TOPIC_CONSUMER
from ScoreLens_FastApi.app.enum import kafka_code
from ScoreLens_FastApi.app.exception.app_exception import AppException, ErrorCode
from ScoreLens_FastApi.app.enum.kafka_code import KafkaCode

import logging
import json

from ScoreLens_FastApi.app.request.kafka_request import ProducerRequest
from ScoreLens_FastApi.app.service.kafka_producer_service import send_to_java

logger = logging.getLogger(__name__)


def consume_partition(partition=0):
    c = consumer()
    tp = TopicPartition(TOPIC_CONSUMER, partition)
    c.assign([tp])
    logger.info(f"Consumer assigned to partition {partition} of topic '{TOPIC_CONSUMER}'.")
    try:
        for message in c:
            process_message(message)
    except KeyboardInterrupt:
        logger.info("Kafka consumer stopped by user.")
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
    finally:
        c.close()
        logger.info("Kafka consumer closed.")

def consume_all_partitions():
    c = consumer()
    c.subscribe([TOPIC_CONSUMER])
    try:
        for message in c:
            process_message(message)
    except KeyboardInterrupt:
        logger.info("Kafka consumer stopped by user.")
    except Exception as e:
        logger.exception(f"Unexpected error in Kafka consumer: {e}")
        raise AppException(
            status_code=500,
            code=ErrorCode.KAFKA_SEND_ERROR,
            message=f"Kafka consumer encountered an error: {e}"
        )
    finally:
        c.close()
        logger.info("Kafka consumer closed.")


def process_message(message):
    try:
        value = message.value

        # Nếu là bytes thì decode sang string
        if isinstance(value, bytes):
            value = value.decode('utf-8')

        # Nếu là string thì parse json
        if isinstance(value, str):
            if not value.strip():
                logger.warning(f"Empty message at offset {message.offset}, skipping.")
                return

            event = json.loads(value)

        # Nếu là dict thì dùng luôn
        elif isinstance(value, dict):
            event = value

        else:
            raise AppException(
                status_code=500,
                code=ErrorCode.CUSTOM_APP_ERROR,
                message=f"Unsupported message type: {type(value)}"
            )

        logger.info(f"[Partition {message.partition}] Offset {message.offset}: {event}")

        # Validate data field
        data = event.get("data")
        if data is None or not str(data).strip():
            logger.warning(f"Message at offset {message.offset} missing or empty data.")
            return

        code_value = event.get("code")
        handle_code_value(code_value)

    except Exception as e:
        logger.error(f"Failed to process message at offset {message.offset}: {e}")
        raise AppException(
            status_code=500,
            code=ErrorCode.CUSTOM_APP_ERROR,
            message=f"Failed to process message: {e}"
        )


# xử lí enum kafka_code
def handle_code_value(code_value):
    match KafkaCode(code_value):
        case KafkaCode.RUNNING:
            send_to_java(ProducerRequest(code=KafkaCode.RUNNING, data="Send heart beat to spring boot"))

        case _:
            print(f"No action defined for: {code_value}")



