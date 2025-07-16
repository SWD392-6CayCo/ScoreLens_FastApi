import os
import logging
from ultralytics import YOLO
import asyncio
import cv2  # Import OpenCV
from typing import Optional, List, Dict, Tuple
from pydantic import BaseModel, Field
from functools import lru_cache

# Cấu hình logging cho module này
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- Định nghĩa các Pydantic Models ---
class Player(BaseModel):
    id: int
    name: str


class Team(BaseModel):
    id: int
    players: List[Player]


class GameSet(BaseModel):
    id: int
    race_to: int


class MatchData(BaseModel):  # Sửa lỗi: MatchData cần kế thừa từ BaseModel
    camera: str
    total_set: int
    sets: List[GameSet]
    teams: List[Team]


class MatchState9Ball(BaseModel):
    code: str
    table_id: str
    mode_id: int
    data: MatchData


# --- Đường dẫn đến file best.pt của bạn ---
YOLOV8_MODEL_PATH = os.getenv("YOLOV8_MODEL_PATH", "ai/best.pt")


# --- Khởi tạo YOLOv8 Model (chỉ một lần duy nhất) ---
@lru_cache(maxsize=1)
def load_yolov8_model():
    """
    Tải model YOLOv8 từ đường dẫn đã cho.
    YOLOv8 sẽ tự động phát hiện và sử dụng CUDA/GPU nếu có và được cấu hình đúng.
    Nếu không tìm thấy GPU, nó sẽ tự động fallback về CPU.
    """
    try:
        model = YOLO(YOLOV8_MODEL_PATH)
        logger.info(f"YOLOv8 model loaded successfully from: {YOLOV8_MODEL_PATH}")
        return model
    except Exception as e:
        logger.error(f"Failed to load YOLOv8 model from {YOLOV8_MODEL_PATH}: {e}")
        raise


# Tải model ngay khi file này được import để sẵn sàng sử dụng.
yolov8_model_instance = None
try:
    yolov8_model_instance = load_yolov8_model()
except Exception:
    logger.critical("Failed to load YOLOv8 model. Detection functions will not work.")

# Dictionary để lưu trữ các tác vụ phát hiện đang hoạt động và sự kiện dừng của chúng.
# Key: table_id, Value: (asyncio.Task, asyncio.Event)
_active_detection_tasks: Dict[str, Tuple[asyncio.Task, asyncio.Event]] = {}


async def detect_state(match_state: MatchState9Ball, stop_event: asyncio.Event):
    """
    Phát hiện trạng thái trận đấu bằng YOLOv8 từ camera URL.

    Args:
        match_state (MatchState9Ball): Đối tượng chứa thông tin trận đấu,
                                       bao gồm URL camera (RTSP stream) và table_id.
        stop_event (asyncio.Event): Một sự kiện để ra hiệu lệnh dừng quá trình phát hiện.
    """
    if yolov8_model_instance is None:
        logger.error("YOLOv8 model is not loaded. Cannot perform detection for table %s.", match_state.table_id)
        return

    table_id = match_state.table_id
    camera_url = match_state.data.camera

    logger.info(f"Starting detection for table ID: {table_id} from camera: {camera_url}")

    try:
        results = yolov8_model_instance.predict(
            source=camera_url,
            stream=True,
            conf=0.25,
            iou=0.7,
            show=True,  # Đặt thành TRUE để hiển thị cửa sổ video
            save=False,
            verbose=False
        )

        for r in results:
            # Kiểm tra xem có tín hiệu dừng chưa
            if stop_event.is_set():
                logger.info(f"Stop signal received for table {table_id}. Stopping detection.")
                break  # Thoát khỏi vòng lặp xử lý khung hình

            if r.boxes:
                for box in r.boxes:
                    cls_id = int(box.cls)
                    class_name = yolov8_model_instance.names[cls_id]
                    confidence = float(box.conf)
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    logger.info(
                        f"Table {table_id}: Detected '{class_name}' with confidence {confidence:.2f} at [{x1},{y1},{x2},{y2}]")

                    # --- Logic xử lý trạng thái trận đấu dựa trên phát hiện ---
                    # Thêm logic nghiệp vụ
                    if class_name == "ball":
                        pass
                    elif class_name == "cue_stick":
                        pass
                    elif class_name == "pocket":
                        pass

            # SỬA ĐỔI: Thêm một asyncio.sleep nhỏ để nhường quyền kiểm soát cho event loop
            # Điều này giúp tác vụ phản hồi nhanh hơn với tín hiệu hủy và Ctrl+C
            await asyncio.sleep(0.001)  # Hoặc asyncio.sleep(0)

    except asyncio.CancelledError:
        # Xử lý khi tác vụ bị hủy (ví dụ: từ hàm stop_detection_for_table)
        logger.info(f"Detection task for table {table_id} was cancelled.")
    except Exception as e:
        logger.error(f"Error during detection for table ID {table_id} from {camera_url}: {e}")
    finally:
        logger.info(f"Detection for table {table_id} finished or stopped.")
        # Đảm bảo xóa tác vụ khỏi danh sách khi nó kết thúc hoặc bị dừng
        if table_id in _active_detection_tasks:
            # Chỉ xóa nếu tác vụ hiện tại là tác vụ đang được theo dõi
            if _active_detection_tasks[table_id][0] == asyncio.current_task():
                del _active_detection_tasks[table_id]
                logger.info(f"Removed detection task for table {table_id} from active tasks.")

        # Đóng tất cả các cửa sổ OpenCV khi tác vụ kết thúc
        cv2.destroyAllWindows()
        logger.info(f"Closed OpenCV windows for table {table_id}.")


async def start_detection_for_table(match_state: MatchState9Ball):
    """
    Khởi tạo hoặc khởi động lại tác vụ phát hiện YOLOv8 cho một table_id cụ thể.
    Nếu đã có tác vụ đang chạy cho table_id này, nó sẽ được dừng trước.
    """
    table_id = match_state.table_id

    if table_id in _active_detection_tasks:
        logger.warning(f"Detection task for table {table_id} is already running. Stopping existing task.")
        await stop_detection_for_table(table_id)
        # Đợi một chút để tác vụ cũ dừng hẳn trước khi khởi tạo tác vụ mới
        await asyncio.sleep(0.5)

    # Tạo một asyncio.Event mới cho tác vụ này
    stop_event = asyncio.Event()
    # Khởi chạy tác vụ detect_state và truyền stop_event vào
    task = asyncio.create_task(detect_state(match_state, stop_event))
    # Lưu trữ tác vụ và sự kiện dừng vào dictionary
    _active_detection_tasks[table_id] = (task, stop_event)
    logger.info(f"Started new detection task for table: {table_id}")
    return task


async def stop_detection_for_table(table_id: str):
    """
    Dừng tác vụ phát hiện YOLOv8 đang chạy cho một table_id cụ thể.
    """
    if table_id in _active_detection_tasks:
        task, stop_event = _active_detection_tasks[table_id]  # Lấy tác vụ và sự kiện

        logger.info(f"Attempting to stop detection for table: {table_id}")
        stop_event.set()  # Đặt sự kiện dừng để vòng lặp trong detect_state có thể thoát

        # Cố gắng hủy tác vụ. Điều này sẽ gây ra asyncio.CancelledError trong detect_state.
        task.cancel()
        try:
            # Đợi tác vụ hoàn thành việc hủy, với một timeout để tránh treo
            await asyncio.wait_for(task, timeout=5.0)  # Đợi tối đa 5 giây
            logger.info(f"Detection task for table {table_id} successfully stopped.")
        except asyncio.CancelledError:
            logger.info(f"Detection task for table {table_id} confirmed as cancelled.")
        except asyncio.TimeoutError:
            logger.warning(f"Detection task for table {table_id} timed out during stop. It might still be running.")
        except Exception as e:
            logger.error(f"Error waiting for detection task for table {table_id} to stop: {e}")
        finally:
            # Xóa tác vụ khỏi dictionary sau khi cố gắng dừng,
            # hoặc nó sẽ tự xóa trong khối finally của detect_state nếu nó kết thúc tự nhiên.
            if table_id in _active_detection_tasks and _active_detection_tasks[table_id][0] == task:
                del _active_detection_tasks[table_id]
                logger.info(f"Removed detection task for table {table_id} from active tasks after stop attempt.")
    else:
        logger.warning(f"No active detection task found for table: {table_id}")
        # Không ném HTTPException ở đây vì đây là hàm nội bộ của module.
        # HTTPException sẽ được xử lý ở tầng API (main.py).


async def stop_all_detection_tasks():
    """
    Dừng tất cả các tác vụ phát hiện đang hoạt động.
    """
    logger.info("Stopping all active detection tasks...")
    # Tạo một bản sao của keys để tránh lỗi khi sửa đổi dictionary trong vòng lặp
    for table_id in list(_active_detection_tasks.keys()):
        await stop_detection_for_table(table_id)
    logger.info("All detection tasks stopped.")
