import os
import cv2
import time
from pathlib import Path

class ScoreAnalyzer:
    def __init__(self, save_dir="logs"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def analyze_shot(self, cue_ball_id, balls_info, collisions, cushions, player_id, game_set_id):
        balls_remaining = [b for b in balls_info if not b["potted"]]
        target_ball = min([i for i in range(1, 10) if i in [balls_info.index(b) for b in balls_remaining]], default=None)

        score_value = any(b["potted"] for b in balls_info)
        is_foul, foul_message = self.check_foul_rule9(cue_ball_id, target_ball, collisions, balls_info, cushions)

        # Kiểm tra nếu potted ball 9
        potted_9 = any((balls_info.index(b) == 9 and b["potted"]) for b in balls_info)

        message = f"Player {player_id} {'potted a ball' if score_value else 'missed'}"
        if potted_9:
            message = f"Player {player_id} wins the set by potting ball 9!"

        result = {
            "code": "LOGGING",
            "data": {
                "level": "easy",
                "type": "score_create",
                "cueBallId": cue_ball_id,
                "balls": balls_info,
                "collisions": collisions,
                "message": message,
                "details": {
                    "playerID": player_id,
                    "gameSetID": game_set_id,
                    "scoreValue": score_value,
                    "isFoul": is_foul,
                    "potted9": potted_9,
                    "isUncertain": False,
                    "message": foul_message
                }
            }
        }
        return result

    @staticmethod
    def check_foul_rule9(self, cue_ball_id, target_ball, collisions, balls_info, cushions):
        if not collisions:
            return True, "No collision detected"

        first_collision = collisions[0]
        if first_collision['ball1'] != cue_ball_id:
            return True, "Cue ball did not hit target ball first"

        if target_ball is not None and first_collision['ball2'] != target_ball:
            return True, f"Cue ball hit ball {first_collision['ball2']} instead of target ball {target_ball}"

        # Nếu không có bi nào potted và không có bi nào chạm băng sau cú chạm đầu
        any_potted = any(b["potted"] for b in balls_info)
        any_cushion_hit = cushions  # boolean

        if not any_potted and not any_cushion_hit:
            return True, "No ball potted and no ball hit cushion after contact"

        return False, "No foul"

    def save_frame(self, frame, prefix="shot"):
        timestamp = int(time.time())
        file_path = self.save_dir / f"{prefix}_{timestamp}.jpg"
        cv2.imwrite(str(file_path), frame)
        return str(file_path)
