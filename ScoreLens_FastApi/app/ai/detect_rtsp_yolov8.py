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
from .score_analyzer import ScoreAnalyzer

# Constants
COLLISION_DISTANCE_THRESHOLD = 30
POSITION_CHANGE_THRESHOLD = 3
CUSHION_MARGIN = 10  # pixels gần mép bàn tính là chạm băng
BALL_STABLE_FRAMES = 5
STABLE_THRESHOLD = 2  # pixel dịch chuyển nhỏ hơn này coi là đứng yên

# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GameState(Enum):
    BREAKING = 0
    PLAYING = 1

class DetectService:
    def __init__(self, rtsp_url, model_path=None, conf_thres=0.5):


        self.ball_history = {}  # ball_id -> deque các vị trí

        self.rtsp_url = rtsp_url
        self.conf_thres = conf_thres
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model_path = model_path or str(Path(__file__).parent / 'best.pt')
        self.model = YOLO(self.model_path)

        if self.device == 'cuda':
            self.model.fuse()
            self.model.model.half()

        logger.info(f"Class names loaded: {self.model.names}")

        self.cap = None
        self.running = False

        self.analyzer = ScoreAnalyzer(save_dir="logs")
        self.ball_positions = {}
        self.prev_positions = {}
        self.collisions = []
        self.cue_ball_id = 0

        self.state = GameState.BREAKING

    def start(self):
        logger.info("🚀 DetectService (YOLOv8) starting...")
        self.running = True
        self.cap = cv2.VideoCapture(self.rtsp_url)
        if not self.cap.isOpened():
            logger.error("❌ Cannot open video stream.")
            return
        Thread(target=self._run, daemon=True).start()

    def stop(self):
        logger.info("🛑 Stopping DetectService...")
        self.running = False
        if self.cap:
            self.cap.release()
        time.sleep(0.5) # đợi nhẹ cho thread thoát
        cv2.destroyAllWindows()  # đảm bảo đóng toàn bộ cửa sổ

    def _run(self):
        start_time = time.time()

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                logger.warning("⚠️ Lost video stream connection.")
                break

            result_frame, balls_info, cushions = self.detect_frame(frame, start_time)

            fps = 1 / (time.time() - start_time)
            cv2.putText(result_frame, f"FPS: {fps:.2f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Detection (YOLOv8)", result_frame)

            if self.state == GameState.BREAKING:
                if self.is_shot_finished():
                    logger.info("📌 Break shot finished. Switching to PLAYING state.")
                    self._reset_tracking()
                    self.state = GameState.PLAYING
                    start_time = time.time()

            elif self.state == GameState.PLAYING:
                potted_count = sum(1 for ball in balls_info if ball["potted"])
                shot_finished = self.is_shot_finished()

                if shot_finished:
                    result_json = self.analyzer.analyze_shot(
                        cue_ball_id=self.cue_ball_id,
                        balls_info=balls_info,
                        collisions=self.collisions,
                        cushions=cushions,
                        player_id=6,
                        game_set_id=82
                    )
                    frame_path = self.analyzer.save_frame(frame)

                    logger.info("🎱 Shot result:\n%s", result_json)
                    logger.info(f"🖼 Frame saved at: {frame_path}")

                    self._reset_tracking()
                    start_time = time.time()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop()
                break

    @staticmethod
    def check_cushion_hit(position, frame_shape):
        cx, cy = position
        width, height = frame_shape[1], frame_shape[0]
        return (cx <= CUSHION_MARGIN or cy <= CUSHION_MARGIN or
                cx >= width - CUSHION_MARGIN or cy >= height - CUSHION_MARGIN)

    def detect_frame(self, frame, start_time):
        results = self.model.predict(source=frame, conf=self.conf_thres, device=self.device, verbose=False)
        balls_info = []
        current_positions = []
        cushions = []

        for r in results:
            for box in r.boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                conf = box.conf.item()
                cls = int(box.cls.item())

                cx, cy = int((xyxy[0] + xyxy[2]) / 2), int((xyxy[1] + xyxy[3]) / 2)
                if cls not in self.ball_history:
                    self.ball_history[cls] = deque(maxlen=BALL_STABLE_FRAMES) # sai lệch 5 pixel
                self.ball_history[cls].append((cx, cy))
                prev_pos = self.ball_positions.get(cls)

                potted = cx < 0 or cy < 0 or cx > frame.shape[1] or cy > frame.shape[0]
                cushion_hit = self.check_cushion_hit((cx, cy), frame.shape)

                if cushion_hit:
                    cushions.append(cls)

                for other_id, other_pos in self.ball_positions.items():
                    if other_id == cls:
                        continue
                    dist = np.linalg.norm(np.array([cx, cy]) - np.array(other_pos))
                    if dist < COLLISION_DISTANCE_THRESHOLD and not any(
                            c for c in self.collisions if c["ball1"] == cls and c["ball2"] == other_id):
                        self.collisions.append({
                            "ball1": cls,
                            "ball2": other_id,
                            "time": round(time.time() - start_time, 2)
                        })

                current_positions.append((cls, (cx, cy)))

                balls_info.append({
                    "id": cls,
                    "start": prev_pos if prev_pos else [cx, cy],
                    "end": [cx, cy],
                    "potted": potted,
                    "cushion_hit": cushion_hit
                })

        self.prev_positions = self.ball_positions.copy()
        self.ball_positions = dict(current_positions)

        return frame, balls_info, cushions

    def is_shot_finished(self):
        if not self.ball_positions:
            return False
        for ball_id, positions in self.ball_history.items():
            if len(positions) < BALL_STABLE_FRAMES:
                return False  # chưa đủ dữ liệu để xét dừng
            total_movement = sum(
                np.linalg.norm(np.array(positions[i]) - np.array(positions[i - 1]))
                for i in range(1, len(positions))
            )
            avg_movement = total_movement / (len(positions) - 1)
            if avg_movement > STABLE_THRESHOLD:
                return False
        return True

    def _reset_tracking(self):
        self.ball_positions.clear()
        self.prev_positions.clear()
        self.collisions.clear()
        self.ball_history.clear()  # Reset history bi

