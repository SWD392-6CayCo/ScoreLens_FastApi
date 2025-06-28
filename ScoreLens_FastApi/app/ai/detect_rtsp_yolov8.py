# import os
# import time
# import cv2
# import torch
# import numpy as np
# import logging
# from collections import deque
# from pathlib import Path
# from threading import Thread
# from ultralytics import YOLO
# from enum import Enum
# from .score_analyzer import ScoreAnalyzer
# from ScoreLens_FastApi.app.state_manager_class.match_state import MatchState
#
# # Constants
# COLLISION_DISTANCE_THRESHOLD = 30
# POSITION_CHANGE_THRESHOLD = 3
# CUSHION_MARGIN = 10  # pixels gần mép bàn tính là chạm băng
# BALL_STABLE_FRAMES = 5
# STABLE_THRESHOLD = 2  # pixel dịch chuyển nhỏ hơn này coi là đứng yên
#
# # Logger setup
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
#
# #khi phá bi thì k detect gì
# class GameState(Enum):
#     BREAKING = 0
#     PLAYING = 1
#
# class DetectService:
#     def __init__(self, rtsp_url, model_path=None, conf_thres=0.5):
#         self.ball_history = {}  # ball_id -> deque các vị trí
#         self.rtsp_url = rtsp_url
#         self.conf_thres = conf_thres
#
#         # Kiểm tra GPU và gán device
#         if not torch.cuda.is_available():
#             logger.warning("⚠️ GPU không khả dụng, đang dùng CPU")
#             self.device = "cpu"
#         else:
#             logger.info(f"✅ GPU OK: {torch.cuda.get_device_name(0)}")
#             self.device = "cuda"
#
#         # Load model YOLO
#         self.model_path = model_path or str(Path(__file__).parent / 'best.pt')
#         self.model = YOLO(self.model_path)
#
#         # Nếu dùng GPU thì fuse + half + move model lên GPU
#         # if self.device == 'cuda':
#         #     self.model.fuse()
#         #     self.model.model.to(self.device)
#
#         logger.info(f"Class names loaded: {self.model.names}")
#         logger.info(f"Model loaded on: {next(self.model.model.parameters()).device}")
#
#         # Các thành phần còn lại
#         self.cap = None
#         self.running = False
#         self.analyzer = ScoreAnalyzer(save_dir="logs")
#         self.ball_positions = {}
#         self.prev_positions = {}
#         self.collisions = []
#         self.cue_ball_id = 0
#         self.state = GameState.BREAKING
#
#     def start(self):
#         logger.info("DetectService (YOLOv8) starting...")
#         self.running = True
#         self.cap = cv2.VideoCapture(self.rtsp_url)
#         if not self.cap.isOpened():
#             logger.error("Cannot open video stream.")
#             return
#         Thread(target=self._run, daemon=True).start()
#
#     def stop(self):
#         logger.info("Stopping DetectService...")
#         self.running = False
#         if self.cap:
#             self.cap.release()
#         time.sleep(0.5) # đợi nhẹ cho thread thoát
#         cv2.destroyAllWindows()  # đảm bảo đóng toàn bộ cửa sổ
#
#     def _run(self):
#         start_time = time.time()
#         # prev_time = 0
#         # target_fps = 15  # Đặt mục tiêu xử lý 15 khung hình/giây
#         while self.running:
#             ret, frame = self.cap.read()
#             if not ret:
#                 logger.warning("Lost video stream connection.")
#                 break
#
#                 # Dùng current_time để tính toán cho chính xác
#                 start_time_of_shot = time.time()  # Thời gian bắt đầu của cú đánh/lượt xử lý
#
#             result_frame, balls_info, cushions = self.detect_frame(frame, start_time)
#
#             fps = 1 / (time.time() - start_time)
#             cv2.putText(result_frame, f"FPS: {fps:.2f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#             cv2.imshow("Detection (YOLOv8)", result_frame)
#
#             if self.state == GameState.BREAKING:
#                 if self.is_shot_finished():
#                     logger.info("Break shot finished. Switching to PLAYING state.")
#                     self._reset_tracking()
#                     self.state = GameState.PLAYING
#                     start_time = time.time()
#
#             elif self.state == GameState.PLAYING:
#                 shot_finished = self.is_shot_finished()
#
#                 if shot_finished:
#                     potted_count = sum(1 for ball in balls_info if ball["potted"])
#
#                     result_json = self.analyzer.analyze_shot(
#                         cue_ball_id=self.cue_ball_id,
#                         balls_info=balls_info,
#                         collisions=self.collisions,
#                         cushions=cushions,
#                         player_id=MatchState.get_current_player_id(),
#                         game_set_id=MatchState.get_game_set_id()
#                     )
#                     frame_path = self.analyzer.save_frame(frame)
#
#                     logger.info("Shot result:\n%s", result_json)
#                     logger.info(f"Frame saved at: {frame_path}")
#
#                     # Kiểm tra số bi ăn được
#                     if potted_count == 0:
#                         # Nếu không ăn điểm → next turn với is_score = False
#                         player_id = MatchState.next_turn(is_score=False)
#                     else:
#                         # Nếu ăn điểm → next turn với is_score = True
#                         player_id = MatchState.next_turn(is_score=True)
#
#                     # Log hoặc gửi thông báo ai sẽ đánh tiếp
#                     print(f"Next turn: Player {player_id}")
#
#                     self._reset_tracking()
#                     start_time = time.time()
#
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 self.stop()
#                 break
#
#     @staticmethod
#     def check_cushion_hit(position, frame_shape):
#         cx, cy = position
#         width, height = frame_shape[1], frame_shape[0]
#         return (cx <= CUSHION_MARGIN or cy <= CUSHION_MARGIN or
#                 cx >= width - CUSHION_MARGIN or cy >= height - CUSHION_MARGIN)
#
#     def detect_frame(self, frame, start_time):
#         results = self.model.predict(source=frame, conf=self.conf_thres, device=self.device, verbose=False)
#         balls_info = []
#         current_positions = []
#         cushions = []
#
#         for r in results:
#             for box in r.boxes:
#                 xyxy = box.xyxy[0].cpu().numpy().astype(int)
#                 conf = box.conf.item()
#                 cls = int(box.cls.item())
#
#                 cx, cy = int((xyxy[0] + xyxy[2]) / 2), int((xyxy[1] + xyxy[3]) / 2)
#                 if cls not in self.ball_history:
#                     self.ball_history[cls] = deque(maxlen=BALL_STABLE_FRAMES) # sai lệch 5 pixel
#                 self.ball_history[cls].append((cx, cy))
#                 prev_pos = self.ball_positions.get(cls)
#
#                 potted = cx < 0 or cy < 0 or cx > frame.shape[1] or cy > frame.shape[0]
#                 cushion_hit = self.check_cushion_hit((cx, cy), frame.shape)
#
#                 if cushion_hit:
#                     cushions.append(cls)
#
#                 for other_id, other_pos in self.ball_positions.items():
#                     if other_id == cls:
#                         continue
#                     dist = np.linalg.norm(np.array([cx, cy]) - np.array(other_pos))
#                     if dist < COLLISION_DISTANCE_THRESHOLD and not any(
#                             c for c in self.collisions if c["ball1"] == cls and c["ball2"] == other_id):
#                         self.collisions.append({
#                             "ball1": cls,
#                             "ball2": other_id,
#                             "time": round(time.time() - start_time, 2)
#                         })
#
#                 current_positions.append((cls, (cx, cy)))
#
#                 balls_info.append({
#                     "id": cls,
#                     "start": prev_pos if prev_pos else [cx, cy],
#                     "end": [cx, cy],
#                     "potted": potted,
#                     "cushion_hit": cushion_hit
#                 })
#
#         self.prev_positions = self.ball_positions.copy()
#         self.ball_positions = dict(current_positions)
#
#         return frame, balls_info, cushions
#
#     def is_shot_finished(self):
#         if not self.ball_positions:
#             return False
#         for ball_id, positions in self.ball_history.items():
#             if len(positions) < BALL_STABLE_FRAMES:
#                 return False  # chưa đủ dữ liệu để xét dừng
#             total_movement = sum(
#                 np.linalg.norm(np.array(positions[i]) - np.array(positions[i - 1]))
#                 for i in range(1, len(positions))
#             )
#             avg_movement = total_movement / (len(positions) - 1)
#             if avg_movement > STABLE_THRESHOLD:
#                 return False
#         return True
#
#     def _reset_tracking(self):
#         self.ball_positions.clear()
#         self.prev_positions.clear()
#         self.collisions.clear()
#         self.ball_history.clear()  # Reset history bi
#
import os
import time
import cv2
import torch
import numpy as np
import logging
from collections import deque
from pathlib import Path
from threading import Thread
from ultralytics import YOLO
from enum import Enum


# --- Giả lập các lớp phụ thuộc ---
class ScoreAnalyzer:
    def __init__(self, save_dir="logs"):
        self.save_dir = Path(save_dir);
        self.save_dir.mkdir(exist_ok=True)

    def analyze_shot(self, **kwargs):
        logger.info("Analyzing shot data: %s", kwargs)
        return {"status": "analyzed"}

    def save_frame(self, frame):
        path = str(self.save_dir / f"frame_{int(time.time())}.jpg");
        cv2.imwrite(path, frame)
        return path


class MatchState:
    _current_player_id = 1

    @staticmethod
    def get_current_player_id(): return MatchState._current_player_id

    @staticmethod
    def next_turn(is_score):
        if not is_score: MatchState._current_player_id = 2 if MatchState._current_player_id == 1 else 1
        return MatchState._current_player_id


# --- Các hằng số ---
COLLISION_DISTANCE_THRESHOLD = 30
CUSHION_MARGIN = 15
BALL_STABLE_FRAMES = 10
STABLE_THRESHOLD = 1.5
POTTED_CONFIRMATION_FRAMES = 5
MOVEMENT_START_THRESHOLD = 10

# --- Cấu hình Logger ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- LỚP MỚI ĐỂ QUẢN LÝ TRẠNG THÁI CỦA MỘT CÚ ĐÁNH ---
class Shot:
    def __init__(self, shot_number, initial_positions):
        self.shot_number = shot_number
        self.initial_positions = initial_positions
        self.collisions = []
        self.cushions = set()
        self.potted = []
        self.moved_balls = set()

    def add_collision(self, ball1, ball2):
        # Sắp xếp để tránh trùng lặp (1,2) và (2,1)
        pair = tuple(sorted((ball1, ball2)))
        if pair not in self.collisions:
            self.collisions.append(pair)

    def get_summary(self):
        return {
            "shot_number": self.shot_number,
            "collisions": self.collisions,
            "cushions": sorted(list(self.cushions)),
            "potted": self.potted,
            "moved_balls": sorted(list(self.moved_balls))
        }

class DetectService:
    def __init__(self, rtsp_url, model_path=None, conf_thres=0.5):
        self.rtsp_url = rtsp_url
        self.conf_thres = conf_thres
        self.ball_history = {}
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device == 'cuda':
            logger.info(f"✅ GPU OK: {torch.cuda.get_device_name(0)}")
        else:
            logger.warning("⚠️ GPU không khả dụng, đang dùng CPU")

        self.model_path = model_path or str(Path(__file__).parent / 'best.pt')
        self.model = YOLO(self.model_path)
        self.model.to(self.device)
        if self.device == 'cuda': self.model.fuse()
        logger.info(f"Model is running on device: {self.model.device}")

        self.cap = None
        self.running = False
        self.analyzer = ScoreAnalyzer(save_dir="logs")

        # Quản lý trạng thái
        self.ball_positions = {}
        self.disappearance_tracker = {}
        self.shot_in_progress = False
        self.shot_count = 0
        self.current_shot = None  # Object để lưu trữ thông tin cú đánh hiện tại

    def start(self):
        logger.info("DetectService (YOLOv8) starting...")
        self.running = True
        self.cap = cv2.VideoCapture(self.rtsp_url)
        if not self.cap.isOpened():
            logger.error(f"Không thể mở luồng RTSP: {self.rtsp_url}");
            return
        Thread(target=self._run, daemon=True).start()

    def stop(self):
        logger.info("Stopping DetectService...");
        self.running = False
        if self.cap: self.cap.release(); time.sleep(0.5); cv2.destroyAllWindows()

    def _run(self):
        while self.running:
            frame_proc_start_time = time.time()
            ret, frame = self.cap.read()
            if not ret:
                logger.warning("Mất kết nối luồng video. Đang thử kết nối lại...")
                time.sleep(2);
                self.cap.release();
                self.cap = cv2.VideoCapture(self.rtsp_url)
                continue

            annotated_frame = self.detect_frame(frame)

            fps = 1 / (time.time() - frame_proc_start_time)
            cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Detection (YOLOv8)", annotated_frame)

            self.handle_game_state(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop();
                break

    def handle_game_state(self, frame):
        """Xử lý logic game."""
        shot_finished = self.is_shot_finished()
        if not shot_finished:
            return

        shot_summary = self.current_shot.get_summary()
        logger.info(f"--- SHOT #{shot_summary['shot_number']} HAS FINISHED ---")
        logger.info(f"Summary: {shot_summary}")

        self.analyzer.analyze_shot(**shot_summary)
        self.analyzer.save_frame(frame)

        player_id = MatchState.next_turn(is_score=(len(shot_summary['potted']) > 0))
        logger.info(f"Next turn: Player {player_id}")

        self._reset_for_next_shot()

    def detect_frame(self, frame):
        """Hàm cốt lõi: Nhận diện và cập nhật trạng thái."""
        results = self.model.predict(source=frame, conf=self.conf_thres, device=self.device, verbose=False)
        annotated_frame = results[0].plot()

        prev_positions = self.ball_positions.copy()
        current_positions = {
            int(b.cls.item()): (int((b.xyxy[0][0] + b.xyxy[0][2]) / 2), int((b.xyxy[0][1] + b.xyxy[0][3]) / 2)) for b in
            results[0].boxes}
        self.ball_positions = current_positions

        for ball_id, pos in current_positions.items():
            if ball_id not in self.ball_history: self.ball_history[ball_id] = deque(maxlen=BALL_STABLE_FRAMES)
            self.ball_history[ball_id].append(pos)

        # Bắt đầu một cú đánh mới
        if not self.shot_in_progress and prev_positions:
            if any(np.linalg.norm(np.array(pos) - np.array(prev_positions.get(ball_id, pos))) > 5 for ball_id, pos in
                   current_positions.items()):
                self.shot_in_progress = True
                self.shot_count += 1
                self.current_shot = Shot(self.shot_count, prev_positions.copy())
                logger.info(f"--- SHOT #{self.current_shot.shot_number} HAS STARTED (Movement Detected) ---")

        # Nếu một cú đánh đang diễn ra, ghi nhận các sự kiện
        if self.shot_in_progress:
            self.update_shot_events(prev_positions, current_positions, frame)

        return annotated_frame

    def update_shot_events(self, prev_positions, current_positions, frame):
        """Ghi nhận tất cả các sự kiện cho cú đánh hiện tại."""
        # Potted
        potted_ids = self.detect_potted_balls(prev_positions.keys(), current_positions.keys())
        for ball_id in potted_ids:
            self.current_shot.potted.append(ball_id)
            self.current_shot.moved_balls.add(ball_id)

        # Moved, Collisions, Cushions
        frame_h, frame_w, _ = frame.shape
        for ball_id, pos in current_positions.items():
            initial_pos = self.current_shot.initial_positions.get(ball_id)
            if initial_pos and np.linalg.norm(np.array(pos) - np.array(initial_pos)) > MOVEMENT_START_THRESHOLD:
                self.current_shot.moved_balls.add(ball_id)

            if self.check_cushion_hit(pos, frame_h, frame_w): self.current_shot.cushions.add(ball_id)

            for other_id, other_pos in current_positions.items():
                if other_id > ball_id and np.linalg.norm(
                        np.array(pos) - np.array(other_pos)) < COLLISION_DISTANCE_THRESHOLD:
                    self.current_shot.add_collision(ball_id, other_id)

    def detect_potted_balls(self, previous_ids, current_ids):
        """Xác định bi vào lỗ và trả về danh sách ID các bi đã được xác nhận."""
        confirmed_potted = []
        disappeared = set(previous_ids) - set(current_ids)
        for ball_id in disappeared:
            self.disappearance_tracker[ball_id] = self.disappearance_tracker.get(ball_id, 0) + 1

        reappeared = set(current_ids) - set(previous_ids)
        for ball_id in reappeared:
            if ball_id in self.disappearance_tracker: del self.disappearance_tracker[ball_id]

        for ball_id, missing_frames in list(self.disappearance_tracker.items()):
            if missing_frames >= POTTED_CONFIRMATION_FRAMES:
                logger.info(f"✅ Ball POTTED (confirmed): {ball_id}")
                confirmed_potted.append(ball_id)
                del self.disappearance_tracker[ball_id]
        return confirmed_potted

    @staticmethod
    def check_cushion_hit(position, frame_h, frame_w):
        cx, cy = position
        return cx <= CUSHION_MARGIN or cy <= CUSHION_MARGIN or cx >= frame_w - CUSHION_MARGIN or cy >= frame_h - CUSHION_MARGIN

    def is_shot_finished(self):
        """Kiểm tra xem một cú đánh đang diễn ra đã kết thúc hay chưa."""
        if not self.shot_in_progress: return False

        for ball_id in self.current_shot.moved_balls:
            if ball_id in self.ball_positions:
                history = self.ball_history.get(ball_id)
                if not history or len(history) < BALL_STABLE_FRAMES: return False

                movement = sum(
                    np.linalg.norm(np.array(history[i]) - np.array(history[i - 1])) for i in range(1, len(history)))
                avg_movement = movement / (len(history) - 1) if len(history) > 1 else 0

                if avg_movement > STABLE_THRESHOLD: return False

        # Thêm một kiểm tra cuối cùng: cú đánh phải kéo dài ít nhất một chút để tránh kết thúc ngay lập tức
        if len(self.current_shot.moved_balls) == 0:
            return False  # Nếu không có bi nào di chuyển, cú đánh chưa thực sự diễn ra

        return True

    def _reset_for_next_shot(self):
        """Reset lại trạng thái để chuẩn bị cho cú đánh tiếp theo."""
        logger.info("--- Resetting for next shot ---")
        self.shot_in_progress = False
        self.current_shot = None
        self.disappearance_tracker.clear()
        # Không cần reset ball_history vì nó tự cập nhật
        # Không cần reset ball_positions vì nó được ghi đè mỗi frame


# --- Phần chạy thử ---
if __name__ == '__main__':
    RTSP_URL = "rtsp://localhost:8554/mystream"
    MODEL_PATH = "best.pt"
    service = DetectService(rtsp_url=RTSP_URL, model_path=MODEL_PATH)
    service.start()
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        service.stop()
        logger.info("Program terminated by user.")
