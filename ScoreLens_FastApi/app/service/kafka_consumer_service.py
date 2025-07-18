from tokenize import String

from ScoreLens_FastApi.app.config.kafka_consumer_config import consumer, TOPIC_CONSUMER, TABLE_ID_KEY
from ScoreLens_FastApi.app.exception.app_exception import AppException, ErrorCode
from ScoreLens_FastApi.app.enum.kafka_code import KafkaCode
from ScoreLens_FastApi.app.config.deps import get_db  # hàm get_db để lấy session
import logging
# Thêm các import cần thiết
from multiprocessing import Process
from ScoreLens_FastApi.app.ai.stream_processor import startDetect # Import hàm từ file đã tạo
from ScoreLens_FastApi.app.state_manager_class.billiards_match_manager import MatchManager
import logging
import json
from ScoreLens_FastApi.app.state_manager_class.match_state import MatchState9Ball
from ScoreLens_FastApi.app.state_manager_class.detect_state import detect_state, start_detection_for_table, \
    stop_detection_for_table
from ScoreLens_FastApi.app.request.message_request import ProducerRequest
from ScoreLens_FastApi.app.service.kafka_producer_service import send_to_java
from ScoreLens_FastApi.app.service.message_service import delete_kafka_message_by_player, delete_kafka_message_by_game_set

logger = logging.getLogger(__name__)

# Trong file kafka_consumer_service.py

# Dictionary để quản lý các trận đấu đang diễn ra cho mỗi bàn
# Key: table_id, Value: đối tượng MatchManager
active_matches = {}

# Dictionary để quản lý các tiến trình xử lý video
detection_processes = {}


async def consume_all_partitions():
    c = consumer()
    try:
        for message in c:
            key = message.key.decode('utf-8') if message.key else None
            value = message.value
            logger.info(f"Received message: Key='{key}', Value='{value}'")
            # lọc msg theo key = table_id
            if key == TABLE_ID_KEY:
                logger.info(f"Processing message for key {TABLE_ID_KEY}: {value}")
                await process_message(message)

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


async def process_message(message):
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

        await handle_code_value(event, table_id)

    except Exception as e:
        logger.error(f"Failed to process message at offset {message.offset}: {e}")
        raise AppException(
            status_code=500,
            code=ErrorCode.CUSTOM_APP_ERROR,
            message=f"Failed to process message: {e}"
        )


# xử lí enum kafka_code
async def handle_code_value(event, table_id):
    code_value = event.get("code")
    data = event.get("data")
    mode_id = event.get("modeID")
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

        # case KafkaCode.START_STREAM:
        #     try:
        #         if mode_id == 2:
        #             # lưu thông tin người chơi
        #             match = MatchState9Ball()
        #             match.set_from_json(event)
        #
        #             print("=== Match state ===")
        #             print(match)
        #
        #             camera_url = match.data.camera_url
        #
        #             print(f"Camera url for table {table_id}: {camera_url}")
        #
        #             await start_detection_for_table(match)
        #             print(f"Detection started for table: {table_id}")
        #
        #     except Exception as e:
        #         print("Error processing message:", e)

        # case KafkaCode.STOP_STREAM:
        #     match = MatchState9Ball()
        #     await stop_detection_for_table(table_id)
        #     match.clear_match_info()  # dùng hàm clear trong MatchState

        case KafkaCode.START_STREAM:
            try:
                # 1. Parse cấu hình từ Kafka message
                match_config = MatchState9Ball()
                match_config.set_from_json(event)

                # 2. Tạo một đối tượng MatchManager mới
                manager = MatchManager(match_config)

                # 3. Lưu manager này lại, liên kết với table_id
                active_matches[table_id] = manager

                logger.info(f"✅ Match manager created and started for table: {table_id}")

                # 4. (Optional) Bắt đầu tiến trình xử lý video
                camera_url = manager.config.data.camera_url
                process = Process(target=startDetect, args=(camera_url, match_config, manager))
                process.start()
                detection_processes[table_id] = process

            except Exception as e:
                logger.error(f"Error processing START_STREAM: {e}", exc_info=True)

        case KafkaCode.STOP_STREAM:
            logger.info(f"🛑 Received STOP signal for table: {table_id}")

            # 1. Dừng tiến trình xử lý video (nếu có)
            if table_id in detection_processes:
                process_to_stop = detection_processes[table_id]
                if process_to_stop.is_alive():
                    logger.info(f"Terminating video process (PID: {process_to_stop.pid})...")
                    process_to_stop.terminate()  # Gửi tín hiệu dừng
                    process_to_stop.join(timeout=5)  # Chờ tối đa 5s để tiến trình kết thúc
                    logger.info("Video process terminated.")
                # Xóa khỏi danh sách quản lý tiến trình
                del detection_processes[table_id]
            else:
                logger.warning(f"No video process found for table {table_id} to stop.")

            # 2. Xóa đối tượng quản lý logic trận đấu (nếu có)
            if table_id in active_matches:
                del active_matches[table_id]
                logger.info(f"Match manager for table {table_id} has been removed.")
            else:
                logger.warning(f"No match manager found for table {table_id} to remove.")

            logger.info(f"Cleanup for table {table_id} is complete.")

            # Bạn có thể clear state của match ở đây nếu cần
            match = MatchState9Ball()
            match.clear_match_info()

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
