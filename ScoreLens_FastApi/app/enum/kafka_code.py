from enum import Enum

class KafkaCode(str, Enum):
    RUNNING = "RUNNING",
    LOGGING = "LOGGING",
    DELETE_PLAYER = "DELETE_PLAYER",
    DELETE_GAME_SET = "DELETE_GAME_SET",
    DELETE_CONFIRM = "DELETE_CONFIRM",