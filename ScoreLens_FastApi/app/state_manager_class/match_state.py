# class MatchState:
#     current_match_info = None
#     game_set_ids = []
#     current_game_set_index = 0
#     current_team_index = 0
#     last_player_per_team = {}
#     last_game_set_winner = None
#
#     @classmethod
#     def set_match_info(cls, info):
#         cls.current_match_info = info
#         cls.game_set_ids = [s["gameSetID"] for s in cls.current_match_info.get("sets", [])]
#         cls.current_game_set_index = 0
#         cls.current_team_index = 0
#         cls.last_game_set_winner = None
#
#         # Khởi tạo last player cho mỗi team
#         cls.last_player_per_team = {}
#         for team in cls.current_match_info["teams"]:
#             cls.last_player_per_team[team["teamID"]] = None
#
#     @classmethod
#     def get_current_team(cls):
#         return cls.current_match_info["teams"][cls.current_team_index]
#
#     @classmethod
#     def get_next_player_id(cls, team_id):
#         team = next(t for t in cls.current_match_info["teams"] if t["teamID"] == team_id)
#         last_player_id = cls.last_player_per_team[team_id]
#         players = team["players"]
#
#         # Tìm player chưa đánh ở lượt gần nhất
#         available_players = [p["playerID"] for p in players if p["playerID"] != last_player_id]
#
#         if available_players:
#             return available_players[0]  # player chưa đánh gần nhất
#         else:
#             return players[0]["playerID"]  # nếu tất cả đã đánh thì chọn lại player đầu tiên
#
#     @classmethod
#     def next_turn(cls, is_score):
#         current_team = cls.get_current_team()
#         current_team_id = current_team["teamID"]
#
#         # Lấy player tiếp theo của team hiện tại
#         current_player_id = cls.get_next_player_id(current_team_id)
#         cls.last_player_per_team[current_team_id] = current_player_id
#
#         if not is_score:
#             # Nếu trượt thì chuyển team
#             cls.current_team_index = (cls.current_team_index + 1) % len(cls.current_match_info["teams"])
#
#         return current_player_id
#
#     @classmethod
#     def get_current_player_id(cls):
#         current_team_id = cls.get_current_team()["teamID"]
#         return cls.get_next_player_id(current_team_id)
#
#     @classmethod
#     def reset_turn(cls):
#         cls.current_team_index = 0
#         for team_id in cls.last_player_per_team:
#             cls.last_player_per_team[team_id] = None
#
#     @classmethod
#     def get_game_set_id(cls):
#         if cls.current_game_set_index < len(cls.game_set_ids):
#             return cls.game_set_ids[cls.current_game_set_index]
#         return None  # nếu hết set
#
#     @classmethod
#     def next_game_set(cls, winning_team_id):
#         cls.last_game_set_winner = winning_team_id
#         cls.current_game_set_index += 1
#
#         if cls.current_game_set_index >= len(cls.game_set_ids):
#             return None  # hết set rồi
#
#         cls.reset_turn()
#         return winning_team_id
#
#     @classmethod
#     def get_last_game_set_winner(cls):
#         return cls.last_game_set_winner
#
#     @classmethod
#     def clear_match_info(cls):
#         cls.current_match_info = None
#         cls.game_set_ids = []
#         cls.current_game_set_index = 0
#         cls.current_team_index = 0
#         cls.last_player_per_team = {}
#         cls.last_game_set_winner = None
from enum import Enum

import logging
# Sử dụng Enum để định nghĩa các trạng thái một cách rõ ràng
class BallType(Enum):
    SOLID = "solids"
    STRIPE = "stripes"
    EIGHT_BALL = "eight_ball"
    CUE_BALL = "cue_ball"


class GameStatus(Enum):
    IN_PROGRESS = "in_progress"
    TEAM_A_WINS = "team_a_wins"
    TEAM_B_WINS = "team_b_wins"



class MatchState8Ball:
    """
    Lớp quản lý trạng thái dành riêng cho game 8 bi, xử lý luật chơi phức tạp.
    """
    current_match_info = None
    teams = {}  # {0: {'id': 0, 'name': 'Team A'}, 1: {'id': 1, 'name': 'Team B'}}

    # Trạng thái trong một ván đấu (game)
    game_status = GameStatus.IN_PROGRESS
    current_team_index = 0
    is_table_open = True
    ball_assignments = {}  # Ví dụ: {0: BallType.SOLID, 1: BallType.STRIPE}
    balls_on_table = set()
    last_player_id = None

    @classmethod
    def set_match_info(cls, info):
        """Khởi tạo thông tin trận đấu và bắt đầu một ván mới."""
        cls.current_match_info = info
        # Đơn giản hóa cấu trúc team để dễ truy cập
        cls.teams = {
            idx: team for idx, team in enumerate(cls.current_match_info.get("teams", []))
        }
        cls.start_new_game()

    @classmethod
    def start_new_game(cls):
        """Reset các trạng thái để bắt đầu một ván 8 bi mới."""
        print("--- STARTING NEW 8-BALL GAME ---")
        cls.game_status = GameStatus.IN_PROGRESS
        # Giả sử đội thắng ván trước sẽ phá bi ván tiếp theo, hoặc đội 0 nếu là ván đầu
        # cls.current_team_index = ...
        cls.is_table_open = True
        cls.ball_assignments = {}
        cls.balls_on_table = set(range(1, 16))  # Bi từ 1 đến 15
        cls.last_player_id = None

    @staticmethod
    def get_ball_type(ball_number):
        """Xác định loại bi dựa trên số."""
        if ball_number is None: return None
        if 1 <= ball_number <= 7: return BallType.SOLID
        if 9 <= ball_number <= 15: return BallType.STRIPE
        if ball_number == 8: return BallType.EIGHT_BALL
        return None

    @classmethod
    def get_current_player_info(cls):
        """Lấy thông tin của người chơi hiện tại."""
        current_team_data = cls.teams[cls.current_team_index]
        team_id = current_team_data['teamID']
        players = current_team_data['players']

        # Logic xoay vòng người chơi trong team
        if len(players) == 1:
            return players[0]

        # Tìm player không phải là người đánh cuối cùng
        available_players = [p for p in players if p['playerID'] != cls.last_player_id]
        return available_players[0] if available_players else players[0]

    @classmethod
    def update_turn(cls, shot_result: dict):
        """
        Hàm cốt lõi: Cập nhật trạng thái game dựa trên kết quả của một cú đánh.
        shot_result = {
            "potted_balls": [3, 10], # Các bi đã vào lỗ
            "is_foul": False,
            "first_ball_hit": 3 # Bi đầu tiên bị bi cái chạm vào
        }
        """
        if cls.game_status != GameStatus.IN_PROGRESS:
            print("Game has already ended.")
            return

        potted_solids = {b for b in shot_result["potted_balls"] if cls.get_ball_type(b) == BallType.SOLID}
        potted_stripes = {b for b in shot_result["potted_balls"] if cls.get_ball_type(b) == BallType.STRIPE}
        potted_8ball = 8 in shot_result["potted_balls"]

        cls.balls_on_table -= set(shot_result["potted_balls"])

        current_player = cls.get_current_player_info()
        cls.last_player_id = current_player['playerID']

        current_team_id = cls.teams[cls.current_team_index]['teamID']
        opponent_team_index = 1 - cls.current_team_index

        # --- 1. XỬ LÝ BI SỐ 8 ---
        if potted_8ball:
            player_ball_type = cls.ball_assignments.get(current_team_id)
            # Kiểm tra xem nhóm bi của người chơi đã hết chưa
            my_balls_left = any(cls.get_ball_type(b) == player_ball_type for b in cls.balls_on_table)

            if shot_result["is_foul"] or (player_ball_type and my_balls_left):
                # Thua nếu phạm lỗi khi đánh bi 8, hoặc đánh bi 8 vào lỗ quá sớm
                print(f"GAME OVER: Team {current_team_id} loses (illegal 8-ball pot).")
                cls.game_status = GameStatus.TEAM_A_WINS if opponent_team_index == 0 else GameStatus.TEAM_B_WINS
            else:
                # Thắng hợp lệ
                print(f"GAME OVER: Team {current_team_id} WINS!")
                cls.game_status = GameStatus.TEAM_A_WINS if cls.current_team_index == 0 else GameStatus.TEAM_B_WINS
            return

        # --- 2. XỬ LÝ PHẠM LỖI ---
        if shot_result["is_foul"]:
            print(f"Foul by Team {current_team_id}. Turn switches.")
            cls.current_team_index = opponent_team_index
            return

        # --- 3. XỬ LÝ BÀN MỞ (OPEN TABLE) ---
        if cls.is_table_open:
            if potted_solids or potted_stripes:
                # Nếu có bi vào lỗ, xác định nhóm bi
                first_potted_type = cls.get_ball_type(shot_result.get("first_ball_hit"))
                if first_potted_type == BallType.SOLID:
                    cls.ball_assignments[current_team_id] = BallType.SOLID
                    cls.ball_assignments[cls.teams[opponent_team_index]['teamID']] = BallType.STRIPE
                    print(f"Table closed. Team {current_team_id} is SOLIDS.")
                elif first_potted_type == BallType.STRIPE:
                    cls.ball_assignments[current_team_id] = BallType.STRIPE
                    cls.ball_assignments[cls.teams[opponent_team_index]['teamID']] = BallType.SOLID
                    print(f"Table closed. Team {current_team_id} is STRIPES.")
                cls.is_table_open = False
                # Người chơi được đi tiếp
            else:
                # Không có bi nào vào lỗ, mất lượt
                print("Open table, no balls potted. Turn switches.")
                cls.current_team_index = opponent_team_index
            return

        # --- 4. XỬ LÝ KHI BÀN ĐÃ XÁC ĐỊNH NHÓM BI ---
        player_ball_type = cls.ball_assignments.get(current_team_id)
        potted_my_balls = any(cls.get_ball_type(b) == player_ball_type for b in shot_result["potted_balls"])

        if potted_my_balls:
            # Đưa được bi của mình vào lỗ -> đi tiếp
            print(f"Team {current_team_id} legally potted a ball. Continues turn.")
        else:
            # Không đưa được bi của mình vào lỗ -> mất lượt
            print(f"Team {current_team_id} did not pot their ball type. Turn switches.")
            cls.current_team_index = opponent_team_index

logger = logging.getLogger(__name__)

class MatchState9Ball:
    """
    Lớp quản lý trạng thái và luật chơi cho chế độ Bida 9 bi
    """
    lowest_ball = 1
    game_over = False
    current_player = None
    players = []
    match_info = {}

    @classmethod
    def set_match_info(cls, info: dict):
        """
        Khởi tạo thông tin trận đấu và người chơi.
        """
        cls.match_info = info
        team_players = []

        # Gộp toàn bộ playerID từ các đội
        for team in info.get("teams", []):
            for player in team.get("players", []):
                team_players.append(player["playerID"])

        cls.players = team_players
        cls.current_player = team_players[0] if team_players else None
        cls.lowest_ball = 1
        cls.game_over = False

        logger.info("✅ 9-BALL MATCH INITIALIZED")
        logger.info(f"🎮 Players: {cls.players}")
        logger.info(f"🎯 First player: {cls.current_player}")
        logger.info(f"🎬 First target ball: {cls.lowest_ball}")

    @classmethod
    def reset(cls, players=None):
        cls.lowest_ball = 1
        cls.game_over = False
        cls.current_player = players[0] if players else cls.players[0]
        if players:
            cls.players = players
        logger.info("🔁 9-BALL GAME RESET")

    @classmethod
    def get_current_game_set_id(cls):
        """Lấy gameSetID từ thông tin trận đấu đã lưu."""
        # Giả định cấu trúc match_info có "sets" là một danh sách
        try:
            return cls.match_info.get("sets", [{}])[0].get("gameSetID", -1)
        except (IndexError, AttributeError):
            return -1

    @classmethod
    def get_current_player(cls):
        return cls.current_player

    @classmethod
    def next_player(cls):
        idx = cls.players.index(cls.current_player)
        cls.current_player = cls.players[(idx + 1) % len(cls.players)]
        return cls.current_player

    @classmethod
    def update_turn(cls, shot_summary: dict):
        """
        Cập nhật trạng thái sau một cú đánh.
        shot_summary = {
            'potted': [ball_ids],
            'collisions': [(ball1, ball2), ...],
            'first_hit': ball_id (int)
        }
        """
        potted = shot_summary.get("potted", [])
        first_hit = shot_summary.get("first_hit")
        cue_ball_id = 0

        is_foul = (first_hit != cls.lowest_ball)
        if cue_ball_id in potted:
            is_foul = True
            logger.info("❌ Foul: Cue ball potted.")

        if 9 in potted:
            if not is_foul:
                cls.game_over = True
                logger.info("🏆 Player wins by pocketing 9 ball!")
            else:
                logger.info("⚠️ 9-ball potted during a foul. It will be respotted.")

        if not cls.game_over:
            if is_foul or not potted:
                cls.next_player()
                logger.info(f"🔄 Turn changes to Player {cls.current_player}")
            else:
                logger.info(f"✅ Player {cls.current_player} keeps turn.")

            # Cập nhật lowest_ball
            remaining = set(range(1, 10)) - set(potted)
            cls.lowest_ball = min(remaining) if remaining else 9
            logger.info(f"🎯 Next lowest ball: {cls.lowest_ball}")

        return {
            "is_foul": is_foul,
            "game_over": cls.game_over,
            "current_player": cls.current_player,
            "lowest_ball": cls.lowest_ball
        }

