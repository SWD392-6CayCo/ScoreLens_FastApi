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
from queue import Queue


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
    def __init__(self):
        pass

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


# --- LỚP ĐỂ QUẢN LÝ TRẠNG THÁI CỦA MỘT CÚ ĐÁNH ---
class Shot:
    def __init__(self, shot_number, initial_positions):
        self.shot_number = shot_number
        self.initial_positions = initial_positions
        self.collisions = []
        self.cushions = set()
        self.potted = []
        self.moved_balls = set()

    def add_collision(self, ball1, ball2):
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


# *** LỚP MỚI: Luồng đọc video chuyên dụng ***
class VideoStream:
    def __init__(self, rtsp_url):
        self.url = rtsp_url
        self.cap = None
        self.queue = Queue(maxsize=2)  # Hàng đợi chỉ chứa 2 frame mới nhất
        self.running = False
        self.thread = Thread(target=self._run, daemon=True)
        logger.info("Video stream thread initialized.")

    def start(self):
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        if not self.cap.isOpened():
            logger.error(f"Cannot open video stream from {self.url}")
            return False

        self.running = True
        self.thread.start()
        logger.info("Video stream thread started.")
        return True

    def _run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                logger.warning("Lost connection, attempting to reconnect...")
                self.cap.release()
                time.sleep(2)
                self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
                continue

            if not self.queue.full():
                self.queue.put(frame)

    def read(self):
        # Lấy frame mới nhất từ hàng đợi
        if self.queue.empty():
            return None
        return self.queue.get()

    def stop(self):
        self.running = False
        self.thread.join()
        if self.cap:
            self.cap.release()
        logger.info("Video stream thread stopped.")


class DetectService:
    def __init__(self, rtsp_url, model_path=None, conf_thres=0.5):
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
        logger.info(f"Class names loaded: {self.model.names}")

        # *** THAY ĐỔI: Sử dụng lớp VideoStream mới ***
        self.video_stream = VideoStream(rtsp_url)

        self.running = False
        self.analyzer = ScoreAnalyzer(save_dir="logs")

        # Quản lý trạng thái
        self.ball_positions = {}
        self.disappearance_tracker = {}
        self.shot_in_progress = False
        self.shot_count = 0
        self.current_shot = None
        self.is_paused = False
        self.cue_ball_id = 14  # 'white'

    def start(self):
        logger.info("DetectService (YOLOv8) starting...")
        if not self.video_stream.start():
            return

        self.running = True
        Thread(target=self._run, daemon=True).start()

    def stop(self):
        logger.info("Stopping DetectService...");
        self.running = False
        self.video_stream.stop()
        time.sleep(0.5)
        cv2.destroyAllWindows()

    def _run(self):
        annotated_frame = None

        # *** THAY ĐỔI: Tạo cửa sổ có thể thay đổi kích thước ***
        cv2.namedWindow("Detection (YOLOv8)", cv2.WINDOW_NORMAL)
        # Đặt kích thước mặc định cho cửa sổ
        cv2.resizeWindow("Detection (YOLOv8)", 1280, 720)

        while self.running:
            if not self.is_paused:
                # *** THAY ĐỔI: Lấy frame từ hàng đợi thay vì đọc trực tiếp ***
                frame = self.video_stream.read()
                if frame is None:
                    time.sleep(0.01)  # Đợi một chút nếu hàng đợi rỗng
                    continue

                frame_proc_start_time = time.time()

                annotated_frame = self.detect_frame(frame)

                fps = 1 / (time.time() - frame_proc_start_time)
                cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                self.handle_game_state(frame)

            if annotated_frame is not None:
                display_frame = annotated_frame.copy()
                if self.is_paused:
                    cv2.putText(display_frame, "-- PAUSED --", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow("Detection (YOLOv8)", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): self.stop(); break
            if key == ord(' '):
                self.is_paused = not self.is_paused
                logger.info("⏸️ Stream paused." if self.is_paused else "▶️ Stream resumed.")

    # Các hàm còn lại giữ nguyên, không cần thay đổi
    def handle_game_state(self, frame):
        shot_finished = self.is_shot_finished()
        if not shot_finished: return

        shot_summary = self.current_shot.get_summary()
        logger.info(f"--- SHOT #{shot_summary['shot_number']} HAS FINISHED ---")
        logger.info(f"Summary: {shot_summary}")

        self.analyzer.analyze_shot(**shot_summary)
        self.analyzer.save_frame(frame)

        player_id = MatchState.next_turn(is_score=(len(shot_summary['potted']) > 0))
        logger.info(f"Next turn: Player {player_id}")

        self._reset_for_next_shot()

    def detect_frame(self, frame):
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

        if not self.shot_in_progress and prev_positions:
            if any(np.linalg.norm(np.array(pos) - np.array(prev_positions.get(ball_id, pos))) > 5 for ball_id, pos in
                   current_positions.items()):
                self.shot_in_progress = True
                self.shot_count += 1
                self.current_shot = Shot(self.shot_count, prev_positions.copy())
                logger.info(f"--- SHOT #{self.current_shot.shot_number} HAS STARTED (Movement Detected) ---")

        if self.shot_in_progress:
            self.update_shot_events(prev_positions, current_positions, frame)

        return annotated_frame

    def update_shot_events(self, prev_positions, current_positions, frame):
        potted_ids = self.detect_potted_balls(prev_positions.keys(), current_positions.keys())
        for ball_id in potted_ids:
            self.current_shot.potted.append(ball_id)
            self.current_shot.moved_balls.add(ball_id)

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
        if not self.shot_in_progress: return False
        if len(self.current_shot.moved_balls) == 0: return False

        for ball_id in self.current_shot.moved_balls:
            if ball_id in self.ball_positions:
                history = self.ball_history.get(ball_id)
                if not history or len(history) < BALL_STABLE_FRAMES: return False

                movement = sum(
                    np.linalg.norm(np.array(history[i]) - np.array(history[i - 1])) for i in range(1, len(history)))
                avg_movement = movement / (len(history) - 1) if len(history) > 1 else 0

                if avg_movement > STABLE_THRESHOLD: return False
        return True

    def _reset_for_next_shot(self):
        logger.info("--- Resetting for next shot ---")
        self.shot_in_progress = False
        self.current_shot = None
        self.disappearance_tracker.clear()


# --- Phần chạy thử ---
if __name__ == '__main__':
    RTSP_URL = "https://scorelens.s3.ap-southeast-2.amazonaws.com/video/output.mp4"
    MODEL_PATH = "best.pt"
    service = DetectService(rtsp_url=RTSP_URL, model_path=MODEL_PATH)
    service.start()
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        service.stop()
        logger.info("Program terminated by user.")
