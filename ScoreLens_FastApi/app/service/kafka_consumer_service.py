
from ScoreLens_FastApi.app.config.kafka_consumer_config import consumer, TOPIC_CONSUMER
from ScoreLens_FastApi.app.exception.app_exception import AppException, ErrorCode
from ScoreLens_FastApi.app.enum.kafka_code import KafkaCode
from ScoreLens_FastApi.app.config.deps import get_db  # hàm get_db để lấy session


import logging
import json

from ScoreLens_FastApi.app.state_manager_class.match_state import MatchState
from ScoreLens_FastApi.app.state_manager_class.detect_state import DetectState
from ScoreLens_FastApi.app.request.message_request import ProducerRequest
from ScoreLens_FastApi.app.service.kafka_producer_service import send_to_java
from ScoreLens_FastApi.app.service.message_service import delete_kafka_message_by_player, delete_kafka_message_by_game_set

logger = logging.getLogger(__name__)


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
        table_id = message.key.decode("utf-8") if message.key else "UNKNOWN"

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

        # Lúc này đảm bảo event là dict
        logger.info(f"[Partition {message.partition}] Offset {message.offset}")
        logger.info(f"[Table {table_id}: {event}")

        # Parse tiếp nested 'data' nếu là string JSON
        data = event.get("data")
        if isinstance(data, str):
            try:
                event["data"] = json.loads(data)
            except json.JSONDecodeError:
                logger.warning(f"Data field is not a valid JSON string at offset {message.offset}, using raw string.")

        # Validate data field
        data = event.get("data")
        if data is None or not str(data).strip():
            logger.warning(f"Message at offset {message.offset} missing or empty data.")
            return

        handle_code_value(event, table_id)

    except Exception as e:
        logger.error(f"Failed to process message at offset {message.offset}: {e}")
        raise AppException(
            status_code=500,
            code=ErrorCode.CUSTOM_APP_ERROR,
            message=f"Failed to process message: {e}"
        )


# xử lí enum kafka_code
def handle_code_value(event, table_id):
    code_value = event.get("code")
    data = event.get("data")
    match KafkaCode(code_value):
        case KafkaCode.RUNNING:
            send_to_java(
                ProducerRequest(code=KafkaCode.RUNNING, tableID=table_id, data="Send heart beat to spring boot"),
                table_id
            )

        case KafkaCode.DELETE_PLAYER:
            player_id = event.get("data")
            with next(get_db()) as db:
                count = delete_kafka_message_by_player(db, player_id)
                send_to_java(ProducerRequest(code=KafkaCode.DELETE_CONFIRM, tableID=table_id, data=count), table_id)

        case KafkaCode.DELETE_GAME_SET:
            game_set_id = event.get("data")
            with next(get_db()) as db:
                count = delete_kafka_message_by_game_set(db,game_set_id)
                send_to_java(ProducerRequest(code=KafkaCode.DELETE_CONFIRM, tableID=table_id, data=count), table_id)

        case KafkaCode.START_STREAM:
            try:
                #lưu thông tin người chơi
                MatchState.set_match_info(table_id, data)

                # Lấy lại và in ra state để kiểm tra
                match_info = MatchState.get_match_info(table_id, {})
                import json

                print("=== Match state ===")
                print(match_info)

                camera_url = match_info.get("camera_url")
                print(f"Camera url for table {table_id}: {camera_url}")

                # bắt đầu stream
                # DetectState.start_detection(camera_url)

            except Exception as e:
                print("Error processing message:", e)

        case KafkaCode.STOP_STREAM:
            MatchState.clear_match_info(table_id)  # dùng hàm clear trong MatchState
            DetectState.stop_detection()
            print("Match info cleared.")

        case _:
            print(f"No action defined for: {code_value}")


def info(data):
    # lấy url camera
    if data and "cameraUrl" in data:
        camera_url = data["cameraUrl"]
        print(camera_url)
    else:
        camera_url = "none"
        print("No camera URL found")


    print("Received match info:", data)

    # # log game sets
    for game_set in data.get("sets", []):
        print(f"Game set ID: {game_set['gameSetID']}")

    # log teams & players
    for team in data["teams"]:
        print(f"Team {team['teamID']}:")
        for player in team["players"]:
            print(f"  Player {player['playerID']} - {player['name']}")
