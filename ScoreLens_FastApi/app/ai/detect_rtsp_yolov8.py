from pathlib import Path

import cv2
from threading import Thread
from ultralytics import YOLO
import torch


class DetectService:
    def __init__(self, rtsp_url, model_path=None, conf_thres=0.5):
        self.rtsp_url = rtsp_url
        self.conf_thres = conf_thres
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model_path = model_path or str(Path(__file__).parent / 'best.pt')
        self.model = YOLO(self.model_path)

        print(f"[DEBUG] Class names loaded: {self.model.names}")
        self.cap = None
        self.running = False

    def start(self):
        print("🚀 DetectService (YOLOv8) starting...")
        self.running = True
        self.cap = cv2.VideoCapture(self.rtsp_url)
        if not self.cap.isOpened():
            print("❌ Không thể mở video stream.")
            return

        t = Thread(target=self._run, daemon=True)
        t.start()

    def stop(self):
        print("🛑 Stopping DetectService...")
        self.running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

    def _run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("⚠️ Mất kết nối video stream.")
                break

            result_frame = self.detect_frame(frame)

            cv2.imshow("Detection (YOLOv8)", result_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop()
                break

    def detect_frame(self, frame):
        results = self.model.predict(source=frame, conf=self.conf_thres, device=self.device, verbose=False)

        for r in results:
            for box in r.boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                conf = box.conf.item()
                cls = int(box.cls.item())
                label = f"{self.model.names[cls]} {conf:.2f}"

                print(f"[DEBUG] Detected {label}")

                cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
                cv2.putText(frame, label, (xyxy[0], xyxy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return frame
