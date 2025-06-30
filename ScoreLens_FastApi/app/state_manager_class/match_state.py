class MatchState:
    current_match_info = None
    game_set_ids = []
    current_game_set_index = 0
    current_team_index = 0
    last_player_per_team = {}
    last_game_set_winner = None

    @classmethod
    def set_match_info(cls, info):
        cls.current_match_info = info
        cls.game_set_ids = [s["gameSetID"] for s in cls.current_match_info.get("sets", [])]
        cls.current_game_set_index = 0
        cls.current_team_index = 0
        cls.last_game_set_winner = None

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

        # Tìm player chưa đánh ở lượt gần nhất
        available_players = [p["playerID"] for p in players if p["playerID"] != last_player_id]

        if available_players:
            return available_players[0]  # player chưa đánh gần nhất
        else:
            return players[0]["playerID"]  # nếu tất cả đã đánh thì chọn lại player đầu tiên

    @classmethod
    def next_turn(cls, is_score):
        current_team = cls.get_current_team()
        current_team_id = current_team["teamID"]

        # Lấy player tiếp theo của team hiện tại
        current_player_id = cls.get_next_player_id(current_team_id)
        cls.last_player_per_team[current_team_id] = current_player_id

        if not is_score:
            # Nếu trượt thì chuyển team
            cls.current_team_index = (cls.current_team_index + 1) % len(cls.current_match_info["teams"])

        return current_player_id

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
        if cls.current_game_set_index < len(cls.game_set_ids):
            return cls.game_set_ids[cls.current_game_set_index]
        return None  # nếu hết set

    @classmethod
    def next_game_set(cls, winning_team_id):
        cls.last_game_set_winner = winning_team_id
        cls.current_game_set_index += 1

        if cls.current_game_set_index >= len(cls.game_set_ids):
            return None  # hết set rồi

        cls.reset_turn()
        return winning_team_id

    @classmethod
    def get_last_game_set_winner(cls):
        return cls.last_game_set_winner

    @classmethod
    def clear_match_info(cls):
        cls.current_match_info = None
        cls.game_set_ids = []
        cls.current_game_set_index = 0
        cls.current_team_index = 0
        cls.last_player_per_team = {}
        cls.last_game_set_winner = None
