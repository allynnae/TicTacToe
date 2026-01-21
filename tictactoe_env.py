from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

WINNING_LINES = (
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),
    (0, 4, 8),
    (2, 4, 6),
)


class TicTacToeEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        invalid_penalty: float = -1.0,
        draw_reward: float = 0.0,
        step_penalty: float = -0.01,
        opponent_first_prob: float = 0.5,
    ) -> None:
        super().__init__()
        self.invalid_penalty = invalid_penalty
        self.draw_reward = draw_reward
        self.step_penalty = step_penalty
        self.opponent_first_prob = opponent_first_prob
        self.agent_mark = 1
        self.opponent_mark = 2

        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(low=0, high=2, shape=(9,), dtype=np.int8)

        self.board = np.zeros(9, dtype=np.int8)

    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        self.board.fill(0)
        info = {}

        if self.np_random.random() < self.opponent_first_prob:
            opp_move = self._opponent_move()
            if opp_move is not None:
                self.board[opp_move] = self.opponent_mark
                info["opponent_started"] = True

        return self._get_obs(), info

    def step(self, action: int):
        info = {}
        terminated = False
        truncated = False

        if self.board[action] != 0:
            return self._get_obs(), self.invalid_penalty, True, truncated, {
                "invalid_action": True
            }

        self.board[action] = self.agent_mark
        winner = self._check_winner()
        if winner == self.agent_mark:
            return self._get_obs(), 1.0, True, truncated, {"result": "win"}
        if self._is_draw():
            return self._get_obs(), self.draw_reward, True, truncated, {"result": "draw"}

        opp_move = self._opponent_move()
        if opp_move is not None:
            self.board[opp_move] = self.opponent_mark

        winner = self._check_winner()
        if winner == self.opponent_mark:
            return self._get_obs(), -1.0, True, truncated, {"result": "loss"}
        if self._is_draw():
            return self._get_obs(), self.draw_reward, True, truncated, {"result": "draw"}

        return self._get_obs(), self.step_penalty, terminated, truncated, info

    def render(self):
        symbols = {0: ".", self.agent_mark: "X", self.opponent_mark: "O"}
        rows = []
        for r in range(3):
            start = r * 3
            rows.append(" ".join(symbols[val] for val in self.board[start : start + 3]))
        print("\n".join(rows))

    def close(self):
        return None

    def _get_obs(self) -> np.ndarray:
        return self.board.astype(np.int8)

    def _check_winner(self) -> int:
        for a, b, c in WINNING_LINES:
            line = (self.board[a], self.board[b], self.board[c])
            if line[0] != 0 and line[0] == line[1] == line[2]:
                return int(line[0])
        return 0

    def _is_draw(self) -> bool:
        return bool(np.all(self.board != 0))

    def _opponent_move(self) -> Optional[int]:
        empties = np.where(self.board == 0)[0].tolist()
        if not empties:
            return None

        for candidate in empties:
            self.board[candidate] = self.opponent_mark
            if self._check_winner() == self.opponent_mark:
                self.board[candidate] = 0
                return candidate
            self.board[candidate] = 0

        for candidate in empties:
            self.board[candidate] = self.agent_mark
            if self._check_winner() == self.agent_mark:
                self.board[candidate] = 0
                return candidate
            self.board[candidate] = 0

        if 4 in empties:
            return 4
        corners = [c for c in (0, 2, 6, 8) if c in empties]
        if corners:
            return int(self.np_random.choice(corners))
        return int(self.np_random.choice(empties))
