from __future__ import annotations
from collections import defaultdict
from typing import Any
import numpy as np
import random

from utils_io import save_pickle, load_pickle


class QLearner:
    def __init__(self, n_actions: int = 3, alpha: float = 0.1, gamma: float = 0.95, seed: int = 0):
        self.Q: defaultdict = defaultdict(lambda: np.zeros(n_actions, dtype=np.float64))
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.rng = random.Random(seed)

    def act(self, state: Any, epsilon: float) -> int:
        if self.rng.random() < epsilon:
            return self.rng.randrange(self.n_actions)
        q = self.Q[state]
        max_q = np.max(q)
        best = [i for i in range(self.n_actions) if q[i] == max_q]
        return self.rng.choice(best)

    def update(self, s: Any, a: int, r: float, s2: Any, done: bool) -> None:
        q_sa = self.Q[s][a]
        target = r if done else (r + self.gamma * float(np.max(self.Q[s2])))
        self.Q[s][a] = (1.0 - self.alpha) * q_sa + self.alpha * target

    @property
    def state_count(self) -> int:
        return len(self.Q)

    def save(self, path: str) -> None:
        data = {k: np.array(v) for k, v in self.Q.items()}
        save_pickle({"n_actions": self.n_actions, "alpha": self.alpha, "gamma": self.gamma, "Q": data}, path)

    @staticmethod
    def load(path: str) -> "QLearner":
        obj = load_pickle(path)
        agent = QLearner(n_actions=obj["n_actions"], alpha=obj["alpha"], gamma=obj["gamma"])
        agent.Q = defaultdict(lambda: np.zeros(agent.n_actions, dtype=np.float64))
        for k, v in obj["Q"].items():
            agent.Q[k] = v
        return agent