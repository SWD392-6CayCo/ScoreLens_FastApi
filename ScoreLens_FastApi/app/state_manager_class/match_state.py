

#lưu thông tin người chơi, dùng MatchState để gọi
class MatchState:
    current_match_info = None
    game_set_id = None
    current_team_index = 0
    last_player_per_team = {}

    @classmethod
    def set_match_info(cls, info):
        cls.current_match_info = info["data"]
        cls.game_set_id = info["data"].get("gameSetID")
        cls.current_team_index = 0

        # Khởi tạo last player cho mỗi team
        cls.last_player_per_team = {}
        for team in cls.current_match_info["teams"]:
            cls.last_player_per_team[team["teamID"]] = None

    @classmethod
    def get_current_team(cls):
        return cls.current_match_info["teams"][cls.current_team_index]

    @classmethod
    def get_next_player_id(cls, team_id):
        team = next(t for t in cls.current_match_info["teams"] if t["teamID"] == team_id)
        last_player_id = cls.last_player_per_team[team_id]
        players = team["players"]

        # Lọc ra danh sách playerID chưa đánh ở lượt gần nhất
        available_players = [p["playerID"] for p in players if p["playerID"] != last_player_id]

        if available_players:
            return available_players[0]  # chọn player chưa đánh
        else:
            return players[0]["playerID"]  # nếu tất cả đã đánh thì chọn player đầu tiên

    @classmethod
    def next_turn(cls, is_score):
        current_team = cls.get_current_team()
        current_team_id = current_team["teamID"]

        # Lấy player tiếp theo trong team hiện tại
        current_player_id = cls.get_next_player_id(current_team_id)
        # Cập nhật người vừa đánh
        cls.last_player_per_team[current_team_id] = current_player_id

        if not is_score:
            # Nếu trượt, chuyển sang team tiếp theo
            cls.current_team_index = (cls.current_team_index + 1) % len(cls.current_match_info["teams"])

        return current_player_id  # trả về player vừa thực hiện

    @classmethod
    def get_current_player_id(cls):
        current_team_id = cls.get_current_team()["teamID"]
        return cls.get_next_player_id(current_team_id)

    @classmethod
    def reset_turn(cls):
        cls.current_team_index = 0
        for team_id in cls.last_player_per_team:
            cls.last_player_per_team[team_id] = None

    @classmethod
    def get_game_set_id(cls):
        return cls.game_set_id

    @classmethod
    def clear_match_info(cls):
        cls.current_match_info = None
        cls.game_set_id = None
        cls.current_team_index = 0
        cls.last_player_per_team = {}



