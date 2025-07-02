class MatchState:
    tables = {}

    @classmethod
    def set_match_info(cls, table_id, info):
        cls.tables[table_id] = {
            "current_match_info": info,
            "camera_url": info.get("cameraUrl"),
            "game_set_ids": [s["gameSetID"] for s in info.get("sets", [])],
            "current_game_set_index": 0,
            "current_team_index": 0,
            "last_player_per_team": {team["teamID"]: None for team in info["teams"]},
            "last_game_set_winner": None
        }

    @classmethod
    def get_match_info(cls, table_id, default=None):
        """
        Trả về toàn bộ match info state của bàn có table_id.
        Nếu không tồn tại thì trả về default (hoặc None nếu không truyền).
        """
        return cls.tables.get(table_id, default)

    @classmethod
    def get_current_team(cls, table_id):
        state = cls.tables[table_id]
        return state["current_match_info"]["teams"][state["current_team_index"]]

    @classmethod
    def get_next_player_id(cls, table_id, team_id):
        state = cls.tables[table_id]
        team = next(t for t in state["current_match_info"]["teams"] if t["teamID"] == team_id)
        last_player_id = state["last_player_per_team"][team_id]
        players = team["players"]

        available_players = [p["playerID"] for p in players if p["playerID"] != last_player_id]

        if available_players:
            return available_players[0]
        else:
            return players[0]["playerID"]

    @classmethod
    def next_turn(cls, table_id, is_score):
        state = cls.tables[table_id]
        current_team = cls.get_current_team(table_id)
        current_team_id = current_team["teamID"]

        current_player_id = cls.get_next_player_id(table_id, current_team_id)
        state["last_player_per_team"][current_team_id] = current_player_id

        if not is_score:
            state["current_team_index"] = (state["current_team_index"] + 1) % len(state["current_match_info"]["teams"])

        return current_player_id

    @classmethod
    def get_current_player_id(cls, table_id):
        current_team_id = cls.get_current_team(table_id)["teamID"]
        return cls.get_next_player_id(table_id, current_team_id)

    @classmethod
    def reset_turn(cls, table_id):
        state = cls.tables[table_id]
        state["current_team_index"] = 0
        for team_id in state["last_player_per_team"]:
            state["last_player_per_team"][team_id] = None

    @classmethod
    def get_game_set_id(cls, table_id):
        state = cls.tables[table_id]
        if state["current_game_set_index"] < len(state["game_set_ids"]):
            return state["game_set_ids"][state["current_game_set_index"]]
        return None

    @classmethod
    def next_game_set(cls, table_id, winning_team_id):
        state = cls.tables[table_id]
        state["last_game_set_winner"] = winning_team_id
        state["current_game_set_index"] += 1

        if state["current_game_set_index"] >= len(state["game_set_ids"]):
            return None  # hết set rồi

        cls.reset_turn(table_id)
        return winning_team_id

    @classmethod
    def get_last_game_set_winner(cls, table_id):
        return cls.tables[table_id]["last_game_set_winner"]

    @classmethod
    def clear_match_info(cls, table_id):
        if table_id in cls.tables:
            del cls.tables[table_id]

    @classmethod
    def clear_all(cls):
        cls.tables.clear()
