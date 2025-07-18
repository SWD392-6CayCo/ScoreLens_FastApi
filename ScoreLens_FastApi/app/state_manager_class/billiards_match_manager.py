import json
import random
from typing import List, Optional


# ==============================================================================
# SECTION 1: CÁC LỚP DỮ LIỆU (DATA CLASSES) - Giữ nguyên từ file của bạn
# ==============================================================================

class Player:
    def __init__(self, player_id: int, name: str):
        self.player_id = player_id
        self.name = name

    def __repr__(self) -> str: return f"Player(id={self.player_id}, name='{self.name}')"


class Team:
    def __init__(self, team_id: int, players: list[Player]):
        self.team_id = team_id
        self.players = players

    def __repr__(self) -> str: return f"Team(id={self.team_id}, players={self.players})"


class GameSet:
    def __init__(self, game_set_id: int, race_to: int):
        self.game_set_id = game_set_id
        self.race_to = race_to

    def __repr__(self) -> str: return f"GameSet(id={self.game_set_id}, race_to={self.race_to})"


class MatchData:
    def __init__(self, camera_url: str, total_set: int, sets: list[GameSet], teams: list[Team]):
        self.camera_url = camera_url
        self.total_set = total_set
        self.sets = sets
        self.teams = teams

    def __repr__(
            self) -> str: return f"MatchData(camera='{self.camera_url}', total_set={self.total_set}, sets={self.sets}, teams={self.teams})"


class MatchState9Ball:
    """Lớp chính quản lý toàn bộ trạng thái cấu hình của trận đấu."""

    def __init__(self):
        self.code: Optional[str] = None
        self.table_id: Optional[str] = None
        self.mode_id: Optional[int] = None
        self.data: Optional[MatchData] = None

    def set_from_json(self, json_input: str | dict):
        if isinstance(json_input, str):
            data = json.loads(json_input)
        else:
            data = json_input
        self.code = data.get("code")
        self.table_id = data.get("tableID")
        self.mode_id = data.get("modeID")
        match_data = data.get("data", {})
        parsed_sets = [GameSet(s.get("gameSetID"), s.get("raceTo")) for s in match_data.get("sets", [])]
        parsed_teams = []
        for t in match_data.get("teams", []):
            players = [Player(p.get("playerID"), p.get("name")) for p in t.get("players", [])]
            parsed_teams.append(Team(t.get("teamID"), players))
        self.data = MatchData(match_data.get("cameraUrl"), match_data.get("totalSet"), parsed_sets, parsed_teams)

    def __repr__(self) -> str:
        return f"MatchState9Ball(code='{self.code}', table_id='{self.table_id}', mode_id={self.mode_id}, data={self.data})"


# ==============================================================================
# SECTION 2: LOGIC XỬ LÝ MỘT SET ĐẤU 9 BI (Nâng cấp từ file logic trước)
# ==============================================================================

class NineBallSetLogic:
    """Quản lý logic cho MỘT set đấu 9 bi (ví dụ: một race-to-5)."""

    def __init__(self, player1_name: str, player2_name: str, race_to: int, initial_breaker: int):
        self.player1_name = player1_name
        self.player2_name = player2_name
        self.race_to = race_to
        self.p1_score = 0
        self.p2_score = 0
        self.winner = None
        self.is_set_over = False
        self.breaker = initial_breaker
        self.current_turn_player = self.breaker
        self.balls_on_table: List[int] = []
        self._start_new_rack()

    def _start_new_rack(self):
        self.balls_on_table = list(range(1, 10))
        self.current_turn_player = self.breaker

    def _switch_turn(self):
        self.current_turn_player = 2 if self.current_turn_player == 1 else 1

    def _award_rack_win(self, player_number: int):
        if player_number == 1:
            self.p1_score += 1
        else:
            self.p2_score += 1

        if self.p1_score >= self.race_to or self.p2_score >= self.race_to:
            self.winner = player_number
            self.is_set_over = True
        else:
            self.breaker = 2 if self.breaker == 1 else 1
            self._start_new_rack()

    def get_status(self):
        return {
            "is_set_over": self.is_set_over,
            "winner_of_set": self.winner,
            "set_score": f"{self.p1_score} - {self.p2_score}",
            "current_turn": self.player1_name if self.current_turn_player == 1 else self.player2_name,
            "balls_on_table": sorted(self.balls_on_table),
        }

    def legal_shot_pocketed(self, ball_number: int):
        if self.is_set_over or ball_number not in self.balls_on_table: return
        self.balls_on_table.remove(ball_number)
        if ball_number == 9: self._award_rack_win(self.current_turn_player)

    def shot_missed(self):
        if self.is_set_over: return
        self._switch_turn()

    def handle_foul(self):
        if self.is_set_over: return
        if 9 not in self.balls_on_table: self.balls_on_table.append(9)
        self._switch_turn()


# ==============================================================================
# SECTION 3: LỚP QUẢN LÝ TRẬN ĐẤU (Lớp điều phối chính)
# ==============================================================================

class MatchManager:
    """
    Lớp quản lý chính, điều phối toàn bộ trận đấu, bao gồm nhiều set đấu.
    Đây là lớp mà hệ thống bên ngoài sẽ tương tác.
    """

    def __init__(self, match_config: MatchState9Ball):
        if not match_config or not match_config.data:
            raise ValueError("Cấu hình trận đấu không hợp lệ hoặc rỗng.")

        self.config = match_config
        self.table_id = match_config.table_id

        # Lấy thông tin người chơi từ cấu hình
        # Giả định team 1 là Player 1, team 2 là Player 2
        self.player1 = self.config.data.teams[0].players[0]
        self.player2 = self.config.data.teams[1].players[0]

        # Trạng thái tổng của cả trận đấu (tính theo set)
        self.p1_sets_won = 0
        self.p2_sets_won = 0
        self.match_winner: Optional[Player] = None
        self.is_match_over = False

        # Trạng thái của set đấu hiện tại
        self.current_set_index = 0
        self.current_set_logic: Optional[NineBallSetLogic] = None

        # Bắt đầu set đấu đầu tiên
        self._start_current_set()

    def _start_current_set(self):
        """Khởi tạo logic cho set đấu hiện tại dựa trên cấu hình."""
        if self.current_set_index >= len(self.config.data.sets):
            print("Tất cả các set đã hoàn thành. Trận đấu kết thúc.")
            self._determine_match_winner()
            return

        current_set_config = self.config.data.sets[self.current_set_index]
        initial_breaker = random.randint(1, 2)  # Chọn ngẫu nhiên người phá bi cho mỗi set mới

        print(f"\n--- BẮT ĐẦU SET {self.current_set_index + 1} (Race to {current_set_config.race_to}) ---")

        self.current_set_logic = NineBallSetLogic(
            player1_name=self.player1.name,
            player2_name=self.player2.name,
            race_to=current_set_config.race_to,
            initial_breaker=initial_breaker
        )

    def _check_and_advance_set(self):
        """Kiểm tra xem set hiện tại đã kết thúc chưa và chuyển sang set mới nếu cần."""
        if self.current_set_logic and self.current_set_logic.is_set_over:
            winner_of_set = self.current_set_logic.winner
            if winner_of_set == 1:
                self.p1_sets_won += 1
                print(f"🏆 {self.player1.name} đã thắng Set {self.current_set_index + 1}!")
            else:
                self.p2_sets_won += 1
                print(f"🏆 {self.player2.name} đã thắng Set {self.current_set_index + 1}!")

            # Chuyển sang set tiếp theo
            self.current_set_index += 1
            self._start_current_set()

    def _determine_match_winner(self):
        """Xác định người thắng trận chung cuộc."""
        if self.p1_sets_won > self.p2_sets_won:
            self.match_winner = self.player1
        elif self.p2_sets_won > self.p1_sets_won:
            self.match_winner = self.player2
        else:  # Hòa
            self.match_winner = None
        self.is_match_over = True
        print(f"\n🎉🎉🎉 TRẬN ĐẤU KẾT THÚC! Tỉ số set: {self.p1_sets_won} - {self.p2_sets_won}")
        if self.match_winner:
            print(f"Người chiến thắng chung cuộc: {self.match_winner.name}")
        else:
            print("Trận đấu có tỉ số hòa!")

    # --- Các hàm public để hệ thống bên ngoài gọi ---

    def get_match_status(self):
        """Lấy trạng thái tổng quan của toàn bộ trận đấu."""
        status = {
            "is_match_over": self.is_match_over,
            "match_winner": self.match_winner.name if self.match_winner else None,
            "match_set_score": f"{self.player1.name} {self.p1_sets_won} - {self.p2_sets_won} {self.player2.name}",
            "current_set_details": None
        }
        if self.current_set_logic:
            status["current_set_details"] = self.current_set_logic.get_status()
        return status

    def pocket_ball(self, ball_number: int):
        if self.is_match_over: return
        self.current_set_logic.legal_shot_pocketed(ball_number)
        self._check_and_advance_set()  # Luôn kiểm tra sau mỗi hành động

    def foul(self):
        if self.is_match_over: return
        self.current_set_logic.handle_foul()
        self._check_and_advance_set()

    def miss(self):
        if self.is_match_over: return
        self.current_set_logic.shot_missed()
        self._check_and_advance_set()


# ==============================================================================
# SECTION 4: VÍ DỤ SỬ DỤNG
# ==============================================================================

if __name__ == '__main__':
    # 1. Giả lập một message JSON từ Kafka
    kafka_event_json = """
    {
        "code": "START_STREAM",
        "tableID": "T01",
        "modeID": 2,
        "data": {
            "cameraUrl": "C:\\Users\ADMIN\Downloads\Demo.mp4",
            "totalSet": 2,
            "sets": [
                { "gameSetID": 101, "raceTo": 3 },
                { "gameSetID": 102, "raceTo": 3 }
            ],
            "teams": [
                { "teamID": 1, "players": [{ "playerID": 1, "name": "Thắng Gió" }] },
                { "teamID": 2, "players": [{ "playerID": 2, "name": "Dũng Con" }] }
            ]
        }
    }
    """

    # 2. Parse thông tin cấu hình trận đấu
    match_config = MatchState9Ball()
    match_config.set_from_json(kafka_event_json)

    # 3. Khởi tạo MatchManager với cấu hình đó
    # Đây là đối tượng bạn sẽ lưu lại cho mỗi table_id
    manager = MatchManager(match_config)

    print("\n=== BẮT ĐẦU MÔ PHỎNG TRẬN ĐẤU ===")
    print("Trạng thái ban đầu:", json.dumps(manager.get_match_status(), indent=2, ensure_ascii=False))

    # --- Mô phỏng SET 1 (Race to 3) ---
    print("\n--- Mô phỏng SET 1 ---")
    # Giả sử Thắng Gió ăn bi 9 và thắng 3 ván liên tiếp
    manager.pocket_ball(9)  # Thắng ván 1
    manager.pocket_ball(9)  # Thắng ván 2
    manager.pocket_ball(9)  # Thắng ván 3 -> Thắng Set 1

    print("Trạng thái sau khi kết thúc Set 1:", json.dumps(manager.get_match_status(), indent=2, ensure_ascii=False))

    # --- Mô phỏng SET 2 (Race to 3) ---
    print("\n--- Mô phỏng SET 2 ---")
    # Giả sử Dũng Con phạm lỗi
    manager.foul()
    # Thắng Gió tận dụng và ăn bi 9, thắng 3 ván
    manager.pocket_ball(9)
    manager.pocket_ball(9)
    manager.pocket_ball(9)  # Thắng Gió thắng Set 2 -> Thắng cả trận

    print("\n=== KẾT THÚC MÔ PHỎNG ===")
    print("Trạng thái cuối cùng:", json.dumps(manager.get_match_status(), indent=2, ensure_ascii=False))