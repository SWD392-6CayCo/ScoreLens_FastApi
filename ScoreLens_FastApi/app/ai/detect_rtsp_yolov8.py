import os
import time
import cv2
import torch
import numpy as np
import logging
from pathlib import Path
from threading import Thread
from ultralytics import YOLO
from .score_analyzer import ScoreAnalyzer

# Constants
COLLISION_DISTANCE_THRESHOLD = 30
POSITION_CHANGE_THRESHOLD = 3
CUSHION_MARGIN = 10  # pixels gần mép bàn tính là chạm băng


# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DetectService:
    def __init__(self, rtsp_url, model_path=None, conf_thres=0.5):
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
        cv2.destroyAllWindows()

    def _run(self):
        start_time = time.time()

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                logger.warning("⚠️ Lost video stream connection.")
                break

            result_frame, balls_info = self.detect_frame(frame, start_time)


            #trả về json
            if self.is_shot_finished():
                cushions = any(ball["cushion_hit"] for ball in balls_info)

                result_json = self.analyzer.analyze_shot(
                    self.cue_ball_id,
                    balls_info,
                    self.collisions,
                    player_id=6,
                    game_set_id=82,
                    cushions=cushions

                )
                frame_path = self.analyzer.save_frame(frame)

                logger.info(f"🎱 Shot result:\n{result_json}")
                logger.info(f"🖼 Frame saved at: {frame_path}")

                self.ball_positions.clear()
                self.prev_positions.clear()
                self.collisions.clear()
                start_time = time.time()

            fps = 1 / (time.time() - start_time)
            cv2.putText(result_frame, f"FPS: {fps:.2f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Detection (YOLOv8)", result_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop()
                break

    @staticmethod
    def check_cushion_hit(self, position, frame_shape):
        cx, cy = position
        width, height = frame_shape[1], frame_shape[0]
        return (
                cx <= CUSHION_MARGIN or cy <= CUSHION_MARGIN or
                cx >= width - CUSHION_MARGIN or cy >= height - CUSHION_MARGIN
        )

    def detect_frame(self, frame, start_time):
        results = self.model.predict(source=frame, conf=self.conf_thres, device=self.device, verbose=False)
        balls_info = []
        current_positions = {}

        for r in results:
            for box in r.boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                conf = box.conf.item()
                cls = int(box.cls.item())
                label = f"{self.model.names[cls]} {conf:.2f}"

                cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
                cv2.putText(frame, label, (xyxy[0], xyxy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cx, cy = int((xyxy[0] + xyxy[2]) / 2), int((xyxy[1] + xyxy[3]) / 2)
                prev_pos = self.ball_positions.get(cls)

                potted = cx < 0 or cy < 0 or cx > frame.shape[1] or cy > frame.shape[0]
                cushion_hit = self.check_cushion_hit((cx, cy), frame.shape)

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

                current_positions[cls] = (cx, cy)

                balls_info.append({
                    "id": cls,
                    "start": prev_pos if prev_pos else [cx, cy],
                    "end": [cx, cy],
                    "potted": potted,
                    "cushion_hit": cushion_hit
                })

        self.prev_positions = self.ball_positions.copy()
        self.ball_positions = current_positions

        return frame, balls_info

    def is_shot_finished(self):
        if not self.ball_positions:
            return False
        for ball_id, pos in self.ball_positions.items():
            prev_pos = self.prev_positions.get(ball_id)
            if not prev_pos:
                return False
            dist = np.linalg.norm(np.array(pos) - np.array(prev_pos))
            if dist > POSITION_CHANGE_THRESHOLD:
                return False
        return True
