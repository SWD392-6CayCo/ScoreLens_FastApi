# Tên file: stream_processor.py
import cv2
import torch
from ultralytics import YOLO
import numpy as np
import json
from ScoreLens_FastApi.app.state_manager_class.billiards_match_manager import MatchState9Ball, MatchManager


# ==============================================================================
# LỚP CỬA SỔ DEBUG (ĐÃ CẬP NHẬT)
# ==============================================================================
class DebugWindow:
    def __init__(self, width=600, height=280):
        self.width = width
        self.height = height
        self.window_name = "Referee Panel"
        self.image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

    def update(self, motion_state: str, balls_on_table: set, all_pocketed_balls: set, last_shot_report: dict,
               current_target_ball: int, shot_count: int):
        """Vẽ lại thông tin debug lên cửa sổ."""
        self.image.fill(0)
        y_pos = 40
        font_scale = 0.9
        font_thickness = 2

        cv2.putText(self.image, f"TONG SO SHOT: {shot_count}", (20, y_pos), cv2.FONT_HERSHEY_DUPLEX, font_scale,
                    (255, 255, 0), font_thickness)
        y_pos += 40

        target_text = f"BI MUC TIEU: {current_target_ball if current_target_ball is not None else 'N/A'}"
        cv2.putText(self.image, target_text, (20, y_pos), cv2.FONT_HERSHEY_DUPLEX, font_scale, (0, 255, 255),
                    font_thickness)
        y_pos += 40

        foul_text = "SHOT CUOI: DANG CHO"
        foul_color = (255, 255, 255)
        if last_shot_report:
            # --- THAY ĐỔI: Cập nhật đường dẫn để đọc cấu trúc JSON mới ---
            is_foul = last_shot_report.get('data', {}).get('details', {}).get('isFoul', False)
            foul_text = f"SHOT CUOI: {'PHAM LOI' if is_foul else 'HOP LE'}"
            foul_color = (0, 0, 255) if is_foul else (0, 255, 0)
        cv2.putText(self.image, foul_text, (20, y_pos), cv2.FONT_HERSHEY_DUPLEX, font_scale, foul_color, font_thickness)
        y_pos += 40

        pocketed_text = f"BI DA LOT: {str(sorted(list(all_pocketed_balls)))}"
        cv2.putText(self.image, pocketed_text, (20, y_pos), cv2.FONT_HERSHEY_DUPLEX, font_scale, (150, 150, 255),
                    font_thickness)
        y_pos += 40

        remaining_text = f"BI CON LAI: {str(sorted(list(balls_on_table)))}"
        cv2.putText(self.image, remaining_text, (20, y_pos), cv2.FONT_HERSHEY_DUPLEX, font_scale, (255, 255, 255),
                    font_thickness)
        y_pos += 40

        state_color = (0, 255, 0) if motion_state == 'STATIONARY' else (0, 165, 255)
        cv2.putText(self.image, f"TRANG THAI: {motion_state}", (20, y_pos), cv2.FONT_HERSHEY_DUPLEX, font_scale,
                    state_color, font_thickness)

        cv2.imshow(self.window_name, self.image)



#
# ==============================================================================
# LỚP SHOTDETECTOR (ĐÃ CẬP NHẬT)
# ==============================================================================
class ShotDetector:
    def __init__(self, stationary_threshold=5, frame_confirmation=50, false_detection_threshold=5):
        # ... (phần khởi tạo giữ nguyên) ...
        self.STATIONARY_THRESHOLD_PIXELS = stationary_threshold
        self.FRAME_CONFIRMATION_COUNT = frame_confirmation
        self.FALSE_DETECTION_THRESHOLD = false_detection_threshold
        self.balls_on_table = set()
        self.last_known_positions = {}
        self.disappeared_counters = {}
        self.new_ball_counters = {}
        self.stationary_frame_counter = 0
        self.motion_state = 'STATIONARY'
        self.shot_info = {"balls_at_shot_start": set()}
        self.last_shot_report = None
        self.all_pocketed_balls = set()
        self.shot_count = 0
        print("✅ ShotDetector initialized.")

    def update_and_detect(self, yolo_results, class_names_map: dict):
        # ... (phần logic của hàm này giữ nguyên) ...
        current_raw_positions = {}
        current_raw_detections = set()
        boxes = yolo_results.boxes.cpu().numpy()
        for box in boxes:
            class_idx = int(box.cls[0])
            if class_idx == 7: continue
            real_ball_name = class_names_map.get(class_idx, "")
            try:
                numeric_part = real_ball_name.split('_')[-1]
                ball_id = int(numeric_part)
                if ball_id not in range(10):
                    continue
                current_raw_detections.add(ball_id)
                x, y, w, h = box.xywh[0]
                current_raw_positions[ball_id] = (x, y)
            except (ValueError, TypeError):
                continue

        disappeared_candidates = self.balls_on_table - current_raw_detections
        for ball_id in disappeared_candidates:
            self.disappeared_counters[ball_id] = self.disappeared_counters.get(ball_id, 0) + 1
            if self.disappeared_counters[ball_id] >= self.FRAME_CONFIRMATION_COUNT:
                self.balls_on_table.remove(ball_id)
                if ball_id in self.last_known_positions: del self.last_known_positions[ball_id]
                del self.disappeared_counters[ball_id]

        reappeared_or_stable_balls = self.balls_on_table.intersection(current_raw_detections)
        for ball_id in reappeared_or_stable_balls:
            self.disappeared_counters[ball_id] = 0
            if ball_id in self.all_pocketed_balls:
                self.all_pocketed_balls.remove(ball_id)
                print(f"Ball {ball_id} reappeared, removed from pocketed balls.")

        newly_detected_balls = current_raw_detections - self.balls_on_table
        confirmed_new_balls = set()
        for ball_id in newly_detected_balls:
            self.new_ball_counters[ball_id] = self.new_ball_counters.get(ball_id, 0) + 1
            if self.new_ball_counters[ball_id] >= self.FALSE_DETECTION_THRESHOLD:
                confirmed_new_balls.add(ball_id)

        self.balls_on_table.update(confirmed_new_balls)

        for ball_id in list(self.new_ball_counters.keys()):
            if ball_id not in newly_detected_balls:
                del self.new_ball_counters[ball_id]

        is_motion_this_frame = False
        if len(self.last_known_positions) > 0:
            for ball_id, pos in current_raw_positions.items():
                if ball_id in self.last_known_positions:
                    dist = np.linalg.norm(np.array(pos) - np.array(self.last_known_positions[ball_id]))
                    if dist > self.STATIONARY_THRESHOLD_PIXELS:
                        is_motion_this_frame = True
                        break

        if self.motion_state == 'STATIONARY':
            if is_motion_this_frame:
                self.motion_state = 'BALLS_MOVING'
                self.stationary_frame_counter = 0
                if self.shot_count == 0:
                    self.shot_info["balls_at_shot_start"] = set(range(10))
                else:
                    self.shot_info["balls_at_shot_start"] = self.balls_on_table.copy()
                self.last_shot_report = None
        elif self.motion_state == 'BALLS_MOVING':
            if not is_motion_this_frame:
                self.stationary_frame_counter += 1
                if self.stationary_frame_counter >= self.FRAME_CONFIRMATION_COUNT:
                    self._analyze_and_report_shot()
                    self.motion_state = 'STATIONARY'
            else:
                self.stationary_frame_counter = 0

        self.last_known_positions = current_raw_positions

    def _analyze_and_report_shot(self):
        """
        Phân tích cú đánh và tạo báo cáo JSON theo định dạng yêu cầu.
        """
        # --- THAY ĐỔI: Toàn bộ logic tạo output được viết lại ---
        print("\n-------------------- SHOT ANALYSIS --------------------")

        # Các giá trị này cần được truyền từ bên ngoài vào trong hệ thống thực tế
        # Ở đây ta dùng giá trị giả định (placeholder)
        table_id_placeholder = "23374e21-2391-41b0-b275-651df88b3b04"
        player_id_placeholder = 290
        gameset_id_placeholder = 242

        # Phân tích các bi
        balls_at_start = self.shot_info["balls_at_shot_start"]
        balls_at_end = self.balls_on_table
        pocketed_balls = balls_at_start - balls_at_end

        for ball_id in pocketed_balls:
            if ball_id not in self.balls_on_table:
                self.all_pocketed_balls.add(ball_id)

        targetable_balls = {b for b in balls_at_start if b != 0}
        target_ball_id = min(targetable_balls) if targetable_balls else -1

        is_foul = 0 in pocketed_balls
        foul_message = "Cue ball pocketed" if is_foul else "No foul"

        # Tạo cấu trúc JSON
        details = {
            "playerID": player_id_placeholder,
            "gameSetID": gameset_id_placeholder,
            "scoreValue": len(pocketed_balls) > 0 and not is_foul,
            "isFoul": is_foul,
            "isUncertain": False,
            "message": foul_message
        }

        data = {
            "cueBallId": 0,
            "targetBallId": target_ball_id,
            "modeID": 2,
            "shotCount": self.shot_count,
            "details": details
        }

        output = {
            "code": "LOGGING",
            "tableID": table_id_placeholder,
            "data": data
        }

        self.last_shot_report = output
        self.shot_count += 1

        # In ra terminal để debug
        print(json.dumps(output, indent=2))
        print("-------------------------------------------------------\n")

# ==============================================================================
# HÀM XỬ LÝ VIDEO CHÍNH (ĐÃ CẬP NHẬT)
# ==============================================================================
def startDetect(rtsp_url: str, match_config: MatchState9Ball, manager: MatchManager):
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    try:
        model = YOLO('best.pt')
        model.to(device)
    except Exception as e:
        print(f"❌ Lỗi tải model: {e}")
        return

    shot_detector = ShotDetector()
    debug_window = DebugWindow()

    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print(f"❌ Lỗi mở video: {rtsp_url}")
        return

    print("✅ Kết nối thành công! Nhấn 'q' để thoát, 'Space' để tạm dừng.")
    cv2.namedWindow("YOLOv8 Detection on RTSP", cv2.WINDOW_NORMAL)

    is_paused = False
    annotated_frame = None

    while True:
        if not is_paused:
            ret, frame = cap.read()
            if not ret: break
            try:
                results = model(frame, device=device, verbose=False)
                class_names_map = model.names
                shot_detector.update_and_detect(results[0], class_names_map)
                annotated_frame = results[0].plot()
            except Exception as e:
                print(f"Lỗi khi xử lý frame: {e}")
                continue

        targetable_balls = {b for b in shot_detector.balls_on_table if b != 0}
        current_target_ball = min(targetable_balls) if targetable_balls else None

        # --- THAY ĐỔI 3: Truyền shot_count vào cửa sổ debug ---
        debug_window.update(
            motion_state=shot_detector.motion_state,
            balls_on_table=shot_detector.balls_on_table,
            all_pocketed_balls=shot_detector.all_pocketed_balls,
            last_shot_report=shot_detector.last_shot_report,
            current_target_ball=current_target_ball,
            shot_count=shot_detector.shot_count  # <--- Tham số mới
        )

        if annotated_frame is not None:
            display_frame = annotated_frame.copy()
            if is_paused:
                (text_width, text_height), _ = cv2.getTextSize("PAUSED", cv2.FONT_HERSHEY_SIMPLEX, 2, 3)
                text_x = (display_frame.shape[1] - text_width) // 2
                text_y = (display_frame.shape[0] + text_height) // 2
                cv2.putText(display_frame, "PAUSED", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3,
                            cv2.LINE_AA)
            cv2.imshow("YOLOv8 Detection on RTSP", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        if key == ord(' '):
            is_paused = not is_paused
            print("Paused." if is_paused else "Resumed.")

    print("🎬 Đang giải phóng tài nguyên...")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    test_url = r"C:\Users\ADMIN\Downloads\one_round.mp4"
    startDetect(test_url)