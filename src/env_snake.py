from __future__ import annotations
from typing import Tuple, Set, Dict, Optional, List
from collections import deque
import random

from config import GameConfig
from features import (UP, RIGHT, DOWN, LEFT,
                       ACT_LEFT, ACT_STRAIGHT, ACT_RIGHT,
                       turn, next_pos, extract_state, _get_level)

Pos = Tuple[int, int]


class SnakeEnv:
    def __init__(self, cfg: GameConfig, seed: int = 0):
        self.cfg = cfg
        self.rng = random.Random(seed)

        self.snake: deque = deque()
        self.body_set: Set[Pos] = set()
        self.dir: int = RIGHT
        self.food: Pos = (0, 0)

        self.static_obstacles: Set[Pos] = set()
        self.temp_obstacles: Dict[Pos, int] = {}

        self.bonus_food: Optional[Pos] = None
        self.bonus_food_timer: int = 0

        self.foods_eaten: int = 0
        self.bonus_eaten: int = 0
        self.steps: int = 0
        self.no_food_steps: int = 0

        self.current_level: int = 0
        self.max_level: int = 0

        self._prev_dist: Optional[int] = None

    def _update_body_set(self):
        self.body_set = set(self.snake)

    def reset(self) -> tuple:
        self.static_obstacles.clear()
        self.temp_obstacles.clear()
        self.bonus_food = None
        self.bonus_food_timer = 0
        self.snake.clear()

        cx = self.cfg.grid_w // 2
        cy = self.cfg.grid_h // 2
        for i in range(4):
            self.snake.append((cx - i, cy))
        self.dir = RIGHT
        self._update_body_set()

        self.foods_eaten = 0
        self.bonus_eaten = 0
        self.steps = 0
        self.no_food_steps = 0
        self.current_level = 0
        self.max_level = 0

        self._spawn_food()
        self._prev_dist = self._manhattan(self.snake[0], self.food)
        return extract_state(self)

    def step(self, action: int) -> Tuple[tuple, float, bool, Dict]:
        self.steps += 1
        prev_level = self.current_level

        self._tick_temp_obstacles()

        if self.bonus_food is not None:
            self.bonus_food_timer -= 1
            if self.bonus_food_timer <= 0:
                self.bonus_food = None

        new_dir = turn(self.dir, action)
        head = self.snake[0]
        new_head = next_pos(head, new_dir)

        will_eat_food = (new_head == self.food)
        will_eat_bonus = (self.bonus_food is not None and new_head == self.bonus_food)

        if self._is_collision(new_head, will_grow=(will_eat_food or will_eat_bonus)):
            reward = self.cfg.reward_death + self.cfg.reward_step
            self._update_body_set()
            return extract_state(self), float(reward), True, self._make_info(win=False, dead=True, reason="collision")

        self.dir = new_dir
        self.snake.appendleft(new_head)

        reward = self.cfg.reward_step
        done = False

        if will_eat_food:
            self.foods_eaten += 1
            self.no_food_steps = 0
            reward += self.cfg.reward_food

            self.current_level = _get_level(self)
            self.max_level = max(self.max_level, self.current_level)

            if self.current_level > prev_level:
                reward += self.cfg.reward_level_up

            if self.foods_eaten >= self.cfg.target_foods:
                self._update_body_set()
                reward += self.cfg.reward_win_bonus
                return extract_state(self), float(reward), True, self._make_info(win=True, dead=False, reason=None)

            self._spawn_food()
            self._prev_dist = self._manhattan(self.snake[0], self.food)
            self._spawn_obstacles_for_level()

            if self.current_level >= 1 and self.bonus_food is None:
                if self.rng.random() < self.cfg.bonus_food_chance:
                    self._spawn_bonus_food()

        elif will_eat_bonus:
            self.bonus_eaten += 1
            self.no_food_steps = 0
            reward += self.cfg.reward_bonus_food
            self.bonus_food = None
            self.bonus_food_timer = 0

        else:
            self.snake.pop()
            self.no_food_steps += 1

            if self.cfg.enable_distance_shaping:
                cur_dist = self._manhattan(self.snake[0], self.food)
                if self._prev_dist is not None:
                    if cur_dist < self._prev_dist:
                        reward += self.cfg.reward_closer
                    elif cur_dist > self._prev_dist:
                        reward += self.cfg.reward_farther
                self._prev_dist = cur_dist

            limit = self.cfg.no_food_limit(self.current_level)
            if self.no_food_steps >= limit:
                self._update_body_set()
                reward += self.cfg.reward_no_food_end
                return extract_state(self), float(reward), True, self._make_info(win=False, dead=False, reason="no_food")

        if self.steps >= self.cfg.max_steps_per_episode:
            done = True

        self._update_body_set()
        return extract_state(self), float(reward), done, self._make_info(win=False, dead=False, reason=None)

    def would_collide(self, pos: Pos) -> bool:
        if not self._in_bounds(pos):
            return True
        if pos in self.static_obstacles:
            return True
        if pos in self.temp_obstacles:
            return True
        tail = self.snake[-1]
        if pos == tail and pos != self.food:
            if self.bonus_food is None or pos != self.bonus_food:
                return False
        return pos in self.body_set

    def _is_collision(self, new_head: Pos, will_grow: bool) -> bool:
        if not self._in_bounds(new_head):
            return True
        if new_head in self.static_obstacles:
            return True
        if new_head in self.temp_obstacles:
            return True
        if will_grow:
            return new_head in self.body_set
        else:
            tail = self.snake[-1]
            if new_head == tail:
                return False
            return new_head in self.body_set

    def _in_bounds(self, pos: Pos) -> bool:
        x, y = pos
        return 0 <= x < self.cfg.grid_w and 0 <= y < self.cfg.grid_h

    def _empty_cells(self, exclude_head_adj: bool = False) -> List[Pos]:
        occupied = self.body_set | self.static_obstacles | set(self.temp_obstacles.keys()) | {self.food}
        if self.bonus_food is not None:
            occupied.add(self.bonus_food)
        empties = [(x, y) for x in range(self.cfg.grid_w)
                   for y in range(self.cfg.grid_h)
                   if (x, y) not in occupied]
        if exclude_head_adj and self.cfg.forbid_obstacle_adjacent_to_head:
            hx, hy = self.snake[0]
            adj = {(hx + 1, hy), (hx - 1, hy), (hx, hy + 1), (hx, hy - 1)}
            empties = [p for p in empties if p not in adj]
        return empties

    def _spawn_food(self) -> None:
        empties = self._empty_cells()
        if empties:
            self.food = self.rng.choice(empties)

    def _spawn_bonus_food(self) -> None:
        empties = self._empty_cells()
        if empties:
            self.bonus_food = self.rng.choice(empties)
            self.bonus_food_timer = self.cfg.bonus_food_timer

    def _spawn_obstacles_for_level(self) -> None:
        if len(self.snake) < self.cfg.obstacle_trigger_length:
            return
        level = self.current_level

        n_static = 0
        if level == 1:
            n_static = self.cfg.obstacles_per_food_lv2
        elif level == 2:
            n_static = self.cfg.obstacles_per_food_lv3

        for _ in range(n_static):
            empties = self._empty_cells(exclude_head_adj=True)
            if empties:
                p = self.rng.choice(empties)
                self.static_obstacles.add(p)

        if level == 2:
            for _ in range(self.cfg.temp_obstacles_per_food_lv3):
                empties = self._empty_cells(exclude_head_adj=True)
                if empties:
                    p = self.rng.choice(empties)
                    self.temp_obstacles[p] = self.cfg.temp_obstacle_lifetime

    def _tick_temp_obstacles(self) -> None:
        expired = [pos for pos, ttl in self.temp_obstacles.items() if ttl <= 1]
        for pos in expired:
            del self.temp_obstacles[pos]
        for pos in self.temp_obstacles:
            self.temp_obstacles[pos] -= 1

    @staticmethod
    def _manhattan(a: Pos, b: Pos) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _make_info(self, win: bool, dead: bool, reason: Optional[str]) -> Dict:
        return {
            "foods": self.foods_eaten,
            "bonus_eaten": self.bonus_eaten,
            "steps": self.steps,
            "no_food_steps": self.no_food_steps,
            "win": win,
            "dead": dead,
            "death_reason": reason,
            "static_obstacles": len(self.static_obstacles),
            "temp_obstacles": len(self.temp_obstacles),
            "level": self.current_level,
            "max_level": self.max_level,
            "snake_length": len(self.snake),
        }
