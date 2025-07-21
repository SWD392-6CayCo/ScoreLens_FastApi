import cv2
import torch
from sqlalchemy import nulls_last
from ultralytics import YOLO
import numpy as np
import json
from ScoreLens_FastApi.app.state_manager_class.billiards_match_manager import MatchState9Ball, MatchManager
from ScoreLens_FastApi.app.service.kafka_producer_service import send_to_java
from ScoreLens_FastApi.app.state_manager_class.detect_state import YOLOV8_MODEL_PATH

# ==============================================================================
# LỚP CỬA SỔ DEBUG
# ==============================================================================
class DebugWindow:
    def __init__(self, width=800, height=480):
        self.width = width
        self.height = height
        self.window_name = "Referee Panel"
        self.image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

    def update(self, motion_state: str, balls_on_table: set, all_pocketed_balls: set, last_shot_report: dict,
               current_target_ball: int, shot_count: int, current_player_id: int, gameset_id: str, race_to_scores: dict):
        """Vẽ lại thông tin debug lên cửa sổ."""
        self.image.fill(0)
        y_pos = 40
        font_scale = 0.9
        font_thickness = 2

        cv2.putText(self.image, f"TONG SO SHOT: {shot_count}", (20, y_pos), cv2.FONT_HERSHEY_DUPLEX, font_scale,
                    (255, 255, 0), font_thickness)
        y_pos += 40

        cv2.putText(self.image, f"LUOT DANH: Player {current_player_id}", (20, y_pos), cv2.FONT_HERSHEY_DUPLEX, font_scale,
                    (255, 150, 0), font_thickness)
        y_pos += 40

        cv2.putText(self.image, f"GAME SET ID: {gameset_id}", (20, y_pos), cv2.FONT_HERSHEY_DUPLEX, font_scale,
                    (255, 0, 255), font_thickness)
        y_pos += 40

        race_to_text = f"RACE TO: {', '.join([f'Player {pid}: {score}' for pid, score in race_to_scores.items()])}"
        cv2.putText(self.image, race_to_text, (20, y_pos), cv2.FONT_HERSHEY_DUPLEX, font_scale, (0, 255, 255),
                    font_thickness)
        y_pos += 40

        target_text = f"BI MUC TIEU: {current_target_ball if current_target_ball is not None else 'N/A'}"
        cv2.putText(self.image, target_text, (20, y_pos), cv2.FONT_HERSHEY_DUPLEX, font_scale, (0, 255, 255),
                    font_thickness)
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


# ==============================================================================
# LỚP SHOTDETECTOR
# ==============================================================================
class ShotDetector:
    def __init__(self, matchInfo: MatchState9Ball, stationary_threshold=5, frame_confirmation=60, false_detection_threshold=5, config_path="config.json"):
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
        self.cue_ball_missing_counter = 0
        self.CUE_BALL_MIN_MISSING_FRAMES = 10  # Thay đổi: Tăng từ 5 lên 10
        self.CUE_BALL_MAX_MISSING_FRAMES = 80  # Thay đổi: Tăng từ 60 lên 80
        try:
            if matchInfo is None:
                raise ValueError("matchInfo is not provided.")

            self.table_id = matchInfo.table_id

            # Lấy danh sách ID người chơi từ các đối tượng Team/Player
            self.player_ids = [player.player_id for team in matchInfo.data.teams for player in team.players]

            # ✅ SỬA Ở ĐÂY: Chuyển từ truy cập dictionary sang truy cập thuộc tính object
            # self.sets giờ sẽ là một list các đối tượng GameSet, không cần tạo lại dict
            self.sets_objects = matchInfo.data.sets  # Đây là list các object GameSet

        except Exception as e:
            # Fallback nếu có lỗi
            print(f"❌ Lỗi đọc config từ matchInfo: {e}. Sử dụng dữ liệu mặc định.")
            self.table_id = "default-table-id"
            self.player_ids = [445, 446]

            # Giả lập lại cấu trúc object để code phía dưới chạy nhất quán
            class MockSet:
                def __init__(self, gid, rt):
                    self.game_set_id = gid
                    self.race_to = rt

            self.sets_objects = [MockSet(331, 2)]

            # ✅ SỬA Ở ĐÂY: Dùng cú pháp truy cập thuộc tính object
        current_set_object = self.sets_objects[self.current_set_index]
        self.gameset_id = current_set_object.game_set_id
        self.race_to_limit = current_set_object.race_to
        self.current_set_index = 0
        self.race_to_scores = {pid: 0 for pid in self.player_ids}
        self.current_player_id = self.player_ids[0]
        self.current_player_index = 0
        self.finished_race = "no"
        print("✅ ShotDetector initialized.")

    def update_and_detect(self, yolo_results, class_names_map: dict):
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

        # Quản lý bộ đếm cho bi cái
        if 0 not in current_raw_detections:
            self.cue_ball_missing_counter += 1
        else:
            self.cue_ball_missing_counter = 0

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

        balls_at_start = self.shot_info["balls_at_shot_start"]
        balls_at_end = self.balls_on_table
        pocketed_balls = balls_at_start - balls_at_end

        # Chỉ thêm các bi từ 1-9 vào all_pocketed_balls
        for ball_id in pocketed_balls:
            if ball_id != 0 and ball_id not in self.balls_on_table:
                self.all_pocketed_balls.add(ball_id)

        targetable_balls = {b for b in balls_at_start if b != 0}
        target_ball_id = min(targetable_balls) if targetable_balls else -1

        # Nếu bi số 9 rớt lỗ, tăng race_to_scores và reset trạng thái bàn
        if ball_id == 9 in pocketed_balls:
            self.race_to_scores[self.current_player_id] += 1
            self.finished_race = "yes"
            print(f"Player {self.current_player_id} wins a race! Current race score: {self.race_to_scores}")
            # Reset trạng thái bàn về ban đầu (tất cả bi 0-9 có mặt)
            self.balls_on_table = set(range(10))
            self.all_pocketed_balls.clear()

        # Chuyển lượt nếu không có bi nào rớt hoặc bi cái mất từ 10-80 frame
        cue_ball_pocketed = self.cue_ball_missing_counter >= self.CUE_BALL_MIN_MISSING_FRAMES and \
                            self.cue_ball_missing_counter <= self.CUE_BALL_MAX_MISSING_FRAMES
        if not pocketed_balls or cue_ball_pocketed:
            self.current_player_index = (self.current_player_index + 1) % len(self.player_ids)
            self.current_player_id = self.player_ids[self.current_player_index]
            if cue_ball_pocketed:
                print(f"Cue ball pocketed (missing for {self.cue_ball_missing_counter} frames), switching turn.")

        # Chuyển sang game set tiếp theo nếu raceTo đạt giới hạn
        if self.race_to_scores[self.current_player_id] >= self.race_to_limit and self.current_set_index < len(self.sets) - 1:
            self.current_set_index += 1
            self.gameset_id = self.sets[self.current_set_index]["gameSetID"]
            self.race_to_limit = self.sets[self.current_set_index]["raceTo"]
            self.race_to_scores = {pid: 0 for pid in self.player_ids}
            self.balls_on_table = set(range(10))
            self.all_pocketed_balls.clear()
            self.finished_race = "no"
            print(f"Chuyển sang game set mới: {self.gameset_id}")

        details = {
            "playerID": self.current_player_id,
            "gameSetID": self.gameset_id,
            "scoreValue": len(pocketed_balls) > 0,
            "isUncertain": False,
            "message": "Shot completed",
            "finished_race": self.finished_race
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
            "tableID": self.table_id,
            "data": data
        }

        self.last_shot_report = output
        self.shot_count += 1

        print(json.dumps(output, indent=2))
        print("-------------------------------------------------------\n")


# ==============================================================================
# HÀM XỬ LÝ VIDEO CHÍNH
# ==============================================================================
def startDetect(rtsp_url: str, match_config: MatchState9Ball = None, manager: MatchManager = None, config_path="config.json"):
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

    shot_detector = ShotDetector(config_path=config_path)
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
                continue

        targetable_balls = {b for b in shot_detector.balls_on_table if b != 0}
        current_target_ball = min(targetable_balls) if targetable_balls else None

        debug_window.update(
            motion_state=shot_detector.motion_state,
            balls_on_table=shot_detector.balls_on_table,
            all_pocketed_balls=shot_detector.all_pocketed_balls,
            last_shot_report=shot_detector.last_shot_report,
            current_target_ball=current_target_ball,
            shot_count=shot_detector.shot_count,
            current_player_id=shot_detector.current_player_id,
            gameset_id=shot_detector.gameset_id,
            race_to_scores=shot_detector.race_to_scores
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
    test_url = r"D:\FPT\Ki7\SWD392\Demo_Fixed_2.mp4"
    startDetect(test_url)