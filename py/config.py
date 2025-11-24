from dataclasses import dataclass

@dataclass(frozen=True)
class GameConfig:
    R1_win: float = 10.0
    R2_correct: float = 1.0
    R3_one_away: float = 0.2
    R4_wrong: float = -0.5
    R5_game_over: float = -2.0
    init_lives: int = 4
