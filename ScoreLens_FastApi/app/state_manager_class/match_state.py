

#lưu thông tin người chơi, dùng MatchState để gọi
class MatchState:
    current_match_info = None
    current_team_index = 0
    current_player_index = 0
    game_set_id = None

    @classmethod
    def set_match_info(cls, info):
        cls.current_match_info = info["data"]
        cls.current_team_index = 0
        cls.current_player_index = 0
        cls.game_set_id = info["data"].get("gameSetID")

    @classmethod
    def get_current_player_id(cls):
        team = cls.current_match_info["teams"][cls.current_team_index]
        player = team["players"][cls.current_player_index]
        return player["playerID"]

    @classmethod
    def next_turn(cls):
        cls.current_team_index = (cls.current_team_index + 1) % len(cls.current_match_info["teams"])
        cls.current_player_index = 0  # Mỗi lượt đổi team thì lấy player đầu tiên

    @classmethod
    def reset_turn(cls):
        cls.current_team_index = 0
        cls.current_player_index = 0

    @classmethod
    def get_game_set_id(cls):
        return cls.game_set_id

    @classmethod
    def clear_match_info(cls):
        cls.current_match_info = None
        cls.current_team_index = 0
        cls.current_player_index = 0
        cls.game_set_id = None


