import json

class Player:
    """Lớp đại diện cho một người chơi."""
    def __init__(self, player_id: int, name: str):
        self.player_id = player_id
        self.name = name

    def __repr__(self) -> str:
        return f"Player(id={self.player_id}, name='{self.name}')"

class Team:
    """Lớp đại diện cho một đội, chứa danh sách người chơi."""
    def __init__(self, team_id: int, players: list[Player]):
        self.team_id = team_id
        self.players = players

    def __repr__(self) -> str:
        return f"Team(id={self.team_id}, players={self.players})"

class GameSet:
    """Lớp đại diện cho thông tin một set đấu."""
    def __init__(self, game_set_id: int, race_to: int):
        self.game_set_id = game_set_id
        self.race_to = race_to

    def __repr__(self) -> str:
        return f"GameSet(id={self.game_set_id}, race_to={self.race_to})"

class MatchData:
    """Lớp chứa dữ liệu chi tiết của trận đấu."""
    def __init__(self, camera_url: str, total_set: int, sets: list[GameSet], teams: list[Team]):
        self.camera_url = camera_url
        self.total_set = total_set
        self.sets = sets
        self.teams = teams

    def __repr__(self) -> str:
        return f"MatchData(camera='{self.camera_url}', total_set={self.total_set}, sets={self.sets}, teams={self.teams})"

class MatchState9Ball:
    """Lớp chính quản lý toàn bộ trạng thái trận đấu 9 bóng."""
    def __init__(self):
        self.code: str | None = None
        self.table_id: str | None = None
        self.mode_id: int | None = None
        self.data: MatchData | None = None

    def set_from_json(self, json_input: str | dict):
        """
        Phân tích dữ liệu từ chuỗi JSON hoặc dictionary và cập nhật trạng thái.

        Args:
            json_input: Chuỗi JSON hoặc dictionary đã được parse.
        """
        if isinstance(json_input, str):
            data = json.loads(json_input)
        else:
            data = json_input

        # Parse các thuộc tính ở cấp cao nhất
        self.code = data.get("code")
        self.table_id = data.get("tableID")
        self.mode_id = data.get("modeID")

        # Parse đối tượng "data" lồng bên trong
        match_data = data.get("data", {})

        # Parse danh sách "sets"
        parsed_sets = [
            GameSet(game_set_id=s.get("gameSetID"), race_to=s.get("raceTo"))
            for s in match_data.get("sets", [])
        ]

        # Parse danh sách "teams"
        parsed_teams = []
        for t in match_data.get("teams", []):
            players = [
                Player(player_id=p.get("playerID"), name=p.get("name"))
                for p in t.get("players", [])
            ]
            team = Team(team_id=t.get("teamID"), players=players)
            parsed_teams.append(team)

        # Tạo đối tượng MatchData
        self.data = MatchData(
            camera_url=match_data.get("cameraUrl"),
            total_set=match_data.get("totalSet"),
            sets=parsed_sets,
            teams=parsed_teams
        )

    def __repr__(self) -> str:
        return (f"MatchState9Ball(code='{self.code}', table_id='{self.table_id}', "
                f"mode_id={self.mode_id}, data={self.data})")

    def clear_match_info(self):
        """
        Xóa tất cả thông tin trận đấu và reset đối tượng về trạng thái ban đầu.
        """
        self.code = None
        self.table_id = None
        self.mode_id = None
        self.data = None
        print("Match info cleared.")