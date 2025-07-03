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
from queue import Queue
from ScoreLens_FastApi.app.state_manager_class.match_state import MatchState9Ball
import json
import threading
from ScoreLens_FastApi.app.api.v1.api_client import KafkaAPIClient


class MatchState:
    _current_player_id = 1

    @staticmethod
    def get_current_player_id(): return MatchState._current_player_id

    @staticmethod
    def next_turn(is_score):
        if not is_score:
            MatchState._current_player_id = 2 if MatchState._current_player_id == 1 else 1
        return MatchState._current_player_id

class ScoreAnalyzer:
    def __init__(self, save_dir="logs"):
        self.save_dir = Path(save_dir);
        self.save_dir.mkdir(exist_ok=True)

    def analyze_shot(self, **kwargs):
        logger.info("Analyzing shot data: %s", kwargs)
        return {"status": "analyzed"}

    def save_frame(self, frame):
        path = str(self.save_dir / f"frame_{int(time.time())}.jpg")
        cv2.imwrite(path, frame)
        return path

class ScoreAnalyzerAPI:
    def __init__(self):
        pass

    def analyze_shot(
        self,
        shot_number: int,
        cue_ball_id: int,
        ball_movements: dict,
        potted_balls: list,
        collisions: list,
        is_foul: bool,
        current_player: int,
        game_set_id: int,
        player_id: int,
        message: str = "No foul"
    ):
        """
        Tổng hợp dữ liệu cú đánh để gửi đi Kafka API
        """

        # Danh sách bi với vị trí đầu và cuối, và tình trạng potted
        balls = []
        for ball_id, movement in ball_movements.items():
            balls.append({
                "start": movement["start"],
                "end": movement["end"],
                "potted": ball_id in potted_balls
            })

        # Danh sách va chạm bi
        collisions_data = []
        for c in collisions:
            collisions_data.append({
                "ball1": c[0],
                "ball2": c[1],
                "time": c[2] if len(c) > 2 else None  # Nếu có timestamp
            })

        shot_data = {
            "code": "LOGGING",
            "data": {
                "level": "easy",  # Có thể để "easy" hoặc truyền vào
                "type": "score_create",
                "cueBallId": cue_ball_id,
                "balls": balls,
                "collisions": collisions_data,
                "message": message,
                "details": {
                    "playerID": player_id,
                    "gameSetID": game_set_id,
                    "scoreValue": not is_foul,
                    "isFoul": is_foul,
                    "isUncertain": False,
                    "message": message
                }
            }
        }

        return shot_data

# --- Hằng số ---
COLLISION_DISTANCE_THRESHOLD = 30
CUSHION_MARGIN = 15
BALL_STABLE_FRAMES = 10
STABLE_THRESHOLD = 1.5
POTTED_CONFIRMATION_FRAMES = 3
MOVEMENT_START_THRESHOLD = 10

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Shot class ---
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

# --- Video reader ---
# class VideoStream:
#     def __init__(self, rtsp_url):
#         self.url = rtsp_url
#         self.cap = None
#         self.queue = Queue(maxsize=2)
#         self.running = False
#         self.thread = Thread(target=self._run, daemon=True)
#
#     def start(self):
#         os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
#         self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
#         if not self.cap.isOpened():
#             logger.error(f"Cannot open video stream from {self.url}")
#             return False
#         self.running = True
#         self.thread.start()
#         return True
#
#     def _run(self):
#         while self.running:
#             ret, frame = self.cap.read()
#             if not ret:
#                 time.sleep(2)
#                 continue
#             if not self.queue.full():
#                 self.queue.put(frame)
#
#     def read(self):
#         if self.queue.empty(): return None
#         return self.queue.get()
#
#     def stop(self):
#         self.running = False
#         self.thread.join()
#         if self.cap: self.cap.release()
class VideoStream:
    def __init__(self, rtsp_url):
        self.url = rtsp_url
        self.cap = None
        self.latest_frame = None
        self.lock = threading.Lock()
        self.running = False
        self.thread = Thread(target=self._run, daemon=True)

    def start(self):
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        if not self.cap.isOpened():
            logger.error(f"Cannot open video stream from {self.url}")
            return False
        self.running = True
        self.thread.start()
        return True

    def _run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(2)
                continue
            with self.lock:
                self.latest_frame = frame

    def read(self):
        with self.lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

    def stop(self):
        self.running = False
        self.thread.join()
        if self.cap:
            self.cap.release()

# --- DetectService ---
class DetectService:
    def __init__(self, rtsp_url, model_path):
        MatchState9Ball.lowest_ball = 2
        # DỜI DÒNG KHỞI TẠO API CLIENT VÀO ĐÂY
        self.api_client = KafkaAPIClient("http://127.0.0.1:8012")
        self.score_api_analyzer = ScoreAnalyzerAPI();
        self.running = False
        self.cue_ball_id = 14
        self.video_stream = VideoStream(rtsp_url)
        self.model = YOLO(model_path)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        if self.device == 'cuda': self.model.fuse()

        self.analyzer = ScoreAnalyzer()

        self.running = False
        self.shot_in_progress = False
        self.shot_count = 0
        self.current_shot = None
        self.ball_positions = {}
        self.ball_history = {}
        self.disappearance_tracker = {}
        self.is_paused = False

    def detect_first_hit_from_collisions(self, collisions: list[tuple[int, int]]) -> int | None:
        for a, b in collisions:
            if self.cue_ball_id in (a, b):
                return b if a == self.cue_ball_id else a
        return None

    def start(self):
        if not self.video_stream.start(): return
        self.running = True
        Thread(target=self._run, daemon=True).start()

    def stop(self):
        self.running = False
        self.video_stream.stop()
        time.sleep(0.5)
        cv2.destroyAllWindows()

    def _run(self):
        cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Detection", 1280, 720)
        while self.running:
            if self.is_paused: continue
            frame = self.video_stream.read()
            if frame is None: continue
            frame = self.detect_frame(frame)
            self.handle_game_state(frame)
            cv2.imshow("Detection", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): self.stop(); break
            if key == ord(' '): self.is_paused = not self.is_paused

    # def detect_frame(self, frame):
    #     results = self.model.predict(source=frame, conf=0.5, device=self.device, verbose=False)
    #     annotated_frame = results[0].plot()
    #
    #     prev = self.ball_positions.copy()
    #     current = {
    #         int(b.cls.item()): (
    #             int((b.xyxy[0][0] + b.xyxy[0][2]) / 2),
    #             int((b.xyxy[0][1] + b.xyxy[0][3]) / 2)) for b in results[0].boxes
    #     }
    #     self.ball_positions = current
    #
    #     for ball_id, pos in current.items():
    #         if ball_id not in self.ball_history:
    #             self.ball_history[ball_id] = deque(maxlen=BALL_STABLE_FRAMES)
    #         self.ball_history[ball_id].append(pos)
    #
    #     if not self.shot_in_progress and prev:
    #         if any(np.linalg.norm(np.array(pos) - np.array(prev.get(ball_id, pos))) > 5 for ball_id, pos in current.items()):
    #             self.shot_in_progress = True
    #             self.shot_count += 1
    #             self.current_shot = Shot(self.shot_count, prev.copy())
    #             logger.info(f"--- SHOT #{self.shot_count} HAS STARTED ---")
    #
    #     if self.shot_in_progress:
    #         self.update_shot_events(prev, current, frame)
    #
    #
    #     return annotated_frame

    def detect_frame(self, frame):
        results = self.model.predict(source=frame, conf=0.5, device=self.device, verbose=False)
        annotated_frame = results[0].plot()

        prev = self.ball_positions.copy()
        current = {
            int(b.cls.item()): (
                int((b.xyxy[0][0] + b.xyxy[0][2]) / 2),
                int((b.xyxy[0][1] + b.xyxy[0][3]) / 2)) for b in results[0].boxes
        }
        self.ball_positions = current

        # Cập nhật lịch sử vị trí bi
        for ball_id, pos in current.items():
            if ball_id not in self.ball_history:
                self.ball_history[ball_id] = deque(maxlen=BALL_STABLE_FRAMES)
            self.ball_history[ball_id].append(pos)

        # Bắt đầu cú đánh nếu phát hiện bi di chuyển
        if not self.shot_in_progress and prev:
            if any(np.linalg.norm(np.array(pos) - np.array(prev.get(ball_id, pos))) > 5 for ball_id, pos in
                   current.items()):
                self.shot_in_progress = True
                self.shot_count += 1
                self.current_shot = Shot(self.shot_count, prev.copy())
                logger.info(f"--- SHOT #{self.shot_count} HAS STARTED ---")

        # Nếu đang trong cú đánh thì tiếp tục cập nhật
        if self.shot_in_progress:
            self.update_shot_events(prev, current, frame)

            # Check xem bi đã dừng hoàn toàn chưa (kết thúc shot)
            if self.is_shot_finished():  # Sử dụng hàm is_shot_finished() đã có để nhất quán
                self.shot_in_progress = False
                logger.info(f"--- SHOT #{self.shot_count} HAS FINISHED ---")

                # =================================================================
                # ▼▼▼ PHẦN LOGIC MỚI ĐỂ GOM VÀ GỬI DỮ LIỆU SHOT ▼▼▼
                # =================================================================

                # 1. Thu thập dữ liệu thô từ cú đánh hiện tại
                final_positions = self.ball_positions.copy()
                potted_balls = self.current_shot.potted
                collisions = self.current_shot.collisions

                # Giả sử bạn có các thông tin này từ MatchState hoặc được truyền vào
                # Ở đây dùng giá trị giả định để minh họa
                # Trong thực tế, bạn sẽ lấy từ MatchState9Ball.get_current_player_info(), v.v.
                player_id = MatchState9Ball.get_current_player();
                game_set_id = MatchState9Ball.get_current_game_set_id()  # Cần có hàm này trong MatchState

                # 2. Xác định lỗi cơ bản (ví dụ: bi cái vào lỗ)
                is_foul = self.cue_ball_id in potted_balls
                message = "Foul: Cue ball potted" if is_foul else "No foul"

                # 3. Chuẩn bị từ điển `ball_movements` theo định dạng yêu cầu
                ball_movements = {}
                all_moved_ball_ids = set(self.current_shot.initial_positions.keys()) | set(final_positions.keys())
                for ball_id in all_moved_ball_ids:
                    ball_movements[ball_id] = {
                        "start": self.current_shot.initial_positions.get(ball_id, [-1, -1]),
                        "end": final_positions.get(ball_id, [-1, -1])
                    }

                # 4. Sử dụng ScoreAnalyzerAPI để tạo payload cuối cùng
                shot_data = self.score_api_analyzer.analyze_shot(
                    shot_number=self.current_shot.shot_number,
                    cue_ball_id=self.cue_ball_id,
                    ball_movements=ball_movements,
                    potted_balls=potted_balls,
                    collisions=collisions,
                    is_foul=is_foul,
                    current_player=player_id,  # API của bạn có vẻ cần cả hai
                    player_id=player_id,
                    game_set_id=game_set_id,
                    message=message
                )

                # 5. Gửi shot log và frame cuối tới API
                if hasattr(self, 'api_client'):
                    logger.info(f"Sending formatted shot data to API: {json.dumps(shot_data, indent=2)}")
                    # Dòng này sẽ gửi dữ liệu đã được định dạng chuẩn
                    self.api_client.send_shot_log(shot_data, frame)

                    # 6. Xử lý logic game và reset cho cú đánh tiếp theo (giữ nguyên)
                # Bạn có thể truyền `shot_data` hoặc `is_foul`, `potted_balls` vào handle_game_state nếu cần
                self.handle_game_state(frame)

                # =================================================================
                # ▲▲▲ KẾT THÚC PHẦN LOGIC MỚI ▲▲▲
                # =================================================================

        return annotated_frame

    def update_shot_events(self, prev, current, frame):
        potted = self.detect_potted_balls(prev.keys(), current.keys())
        for ball_id in potted:
            self.current_shot.potted.append(ball_id)
            self.current_shot.moved_balls.add(ball_id)


        h, w, _ = frame.shape
        for ball_id, pos in current.items():
            init = self.current_shot.initial_positions.get(ball_id)
            if init and np.linalg.norm(np.array(pos) - np.array(init)) > MOVEMENT_START_THRESHOLD:
                self.current_shot.moved_balls.add(ball_id)

            if self.check_cushion_hit(pos, h, w):
                self.current_shot.cushions.add(ball_id)

            for other_id, other_pos in current.items():
                if other_id > ball_id and np.linalg.norm(np.array(pos) - np.array(other_pos)) < COLLISION_DISTANCE_THRESHOLD:
                    self.current_shot.add_collision(ball_id, other_id)

    def detect_potted_balls(self, prev_ids, curr_ids):
        disappeared = set(prev_ids) - set(curr_ids)
        logger.debug(f"🟡 Disappeared balls: {disappeared}")
        logger.debug(f"➡️  Disappearance tracker before: {self.disappearance_tracker}")

        for ball_id in disappeared:
            self.disappearance_tracker[ball_id] = self.disappearance_tracker.get(ball_id, 0) + 1

        reappeared = set(curr_ids) - set(prev_ids)
        for ball_id in reappeared:
            if ball_id in self.disappearance_tracker:
                logger.debug(f"⚠️ Ball {ball_id} reappeared, resetting tracker.")
                del self.disappearance_tracker[ball_id]

        logger.debug(f"➡️  Disappearance tracker after: {self.disappearance_tracker}")

        confirmed = []
        for ball_id, missing in list(self.disappearance_tracker.items()):
            if missing >= POTTED_CONFIRMATION_FRAMES:
                logger.info(f"✅ Ball POTTED: {ball_id} (missing {missing} frames)")
                confirmed.append(ball_id)
                del self.disappearance_tracker[ball_id]

        return confirmed

    def is_shot_finished(self):
        if not self.shot_in_progress or len(self.current_shot.moved_balls) == 0: return False
        for ball_id in self.current_shot.moved_balls:
            if ball_id in self.ball_positions:
                history = self.ball_history.get(ball_id)
                if not history or len(history) < BALL_STABLE_FRAMES: return False
                movement = sum(np.linalg.norm(np.array(history[i]) - np.array(history[i - 1])) for i in range(1, len(history)))
                avg = movement / (len(history) - 1)
                if avg > STABLE_THRESHOLD: return False
        return True

    # def handle_game_state(self, frame):
    #     if not self.is_shot_finished(): return
    #     summary = self.current_shot.get_summary()
    #     logger.info(f"--- SHOT #{summary['shot_number']} HAS FINISHED ---")
    #     logger.info(f"Summary: {summary}")
    #     self.analyzer.analyze_shot(**summary)
    #     self.analyzer.save_frame(frame)
    #     next_player = MatchState.next_turn(is_score=(len(summary['potted']) > 0))
    #     logger.info(f"Next turn: Player {next_player}")
    #     self._reset_for_next_shot()
    def handle_game_state(self, frame):
        if not self.is_shot_finished(): return
        summary = self.current_shot.get_summary()

        # Xác định bi đầu tiên bị chạm
        # first_hit = self.detect_first_hit_from_collisions(summary["collisions"])
        first_hit = self.detect_first_hit_from_collisions(summary["collisions"])
        summary["first_hit"] = first_hit

        logger.info(f"💥 First ball hit: {first_hit}")
        logger.info(f"🎯 Lowest ball (must hit): {MatchState9Ball.lowest_ball}")
        logger.info(f"--- SHOT #{summary['shot_number']} HAS FINISHED ---")
        logger.info(f"Summary: {summary}")

        game_state = MatchState9Ball.update_turn(summary)

        # Lưu log
        self.analyzer.analyze_shot(**summary, **game_state)
        self.analyzer.save_frame(frame)

        if game_state["game_over"]:
            logger.info("🎉 Game over.")
            self.stop()
            return

        next_player = MatchState.next_turn(is_score=(len(summary['potted']) > 0 and not game_state["is_foul"]))
        # logger.info(f"Next turn: Player {next_player}")
        self._reset_for_next_shot()

    def _reset_for_next_shot(self):
        self.shot_in_progress = False
        self.current_shot = None
        self.disappearance_tracker.clear()

    @staticmethod
    def check_cushion_hit(pos, h, w):
        cx, cy = pos
        return cx <= CUSHION_MARGIN or cy <= CUSHION_MARGIN or cx >= w - CUSHION_MARGIN or cy >= h - CUSHION_MARGIN

# --- Main ---
if __name__ == '__main__':
    # Đọc dữ liệu trận đấu từ file JSON
    with open("match_info.json", "r") as f:
        match_info = json.load(f)

    # Thiết lập thông tin trận đấu cho MatchState9Ball
    MatchState9Ball.set_match_info(match_info["data"])

    # Lấy RTSP từ match info
    RTSP_URL = match_info["data"].get("cameraUrl", "rtsp://localhost:8554/mystream")
    MODEL_PATH = "best.pt"

    # Khởi tạo dịch vụ detect
    service = DetectService(rtsp_url=RTSP_URL, model_path=MODEL_PATH)
    service.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        service.stop()
        logger.info("Stopped by user.")
