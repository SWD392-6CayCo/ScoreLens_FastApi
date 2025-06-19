from ScoreLens_FastApi.app.config.kafka_consumer_config import consumer_config

def consume_messages():
    consumer = consumer_config()
    try:
        for message in consumer:
            event = message.value
            print(f"[Partition {message.partition}] Offset {message.offset}: {event}")
    except KeyboardInterrupt:
        print("Consumer stopped.")
    finally:
        consumer.close()
        print("Consumer closed.")
