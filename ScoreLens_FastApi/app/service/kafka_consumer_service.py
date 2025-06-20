from ScoreLens_FastApi.app.config.kafka_consumer_config import consumer_config
from ScoreLens_FastApi.app.exception.app_exception import AppException
from ScoreLens_FastApi.app.exception.app_exception import ErrorCode

import logging

logger = logging.getLogger(__name__)

def consume_messages():
    consumer = consumer_config()
    try:
        for message in consumer:
            try:
                event = message.value
                logger.info(f"[Partition {message.partition}] Offset {message.offset}: {event}")
                # Xử lý message




            except Exception as e:
                logger.error(f"Failed to process message at offset {message.offset}: {e}")
                raise AppException(
                    status_code=500,
                    code=ErrorCode.CUSTOM_APP_ERROR,
                    message=f"Failed to process message: {e}"
                )
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
        consumer.close()
        logger.info("Kafka consumer closed.")
