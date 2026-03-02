from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class GameConfig:
    grid_w: int = 30
    grid_h: int = 30

    target_foods: int = 20                  # 25->20 (更可达)

    level2_threshold: int = 7
    level3_threshold: int = 14

    # Obstacles — 大幅减少
    obstacle_trigger_length: int = 12       # 10->12
    obstacles_per_food_lv2: int = 1         # 2->1
    obstacles_per_food_lv3: int = 1         # 2->1
    temp_obstacles_per_food_lv3: int = 2    # 3->2
    temp_obstacle_lifetime: int = 35
    forbid_obstacle_adjacent_to_head: bool = True

    # Bonus food
    bonus_food_chance: float = 0.5
    bonus_food_timer: int = 80              # 70->80

    # Episode control
    no_food_limit_lv1: int = 500            # 400->500
    no_food_limit_lv2: int = 600            # 500->600
    no_food_limit_lv3: int = 800            # 700->800
    max_steps_per_episode: int = 18000

    # Rewards
    reward_food: float = 10.0
    reward_bonus_food: float = 20.0
    reward_win_bonus: float = 50.0
    reward_level_up: float = 15.0
    reward_death: float = -12.0             # -15->-12 (鼓励探索)
    reward_step: float = -0.02              # -0.03->-0.02 (步数多别罚太重)
    reward_no_food_end: float = -5.0

    # 强距离塑形 — 这是 30x30 的关键
    enable_distance_shaping: bool = True
    reward_closer: float = 0.6             # 0.2->0.6 (强力引导!)
    reward_farther: float = -0.6           # 同上

    def no_food_limit(self, level: int) -> int:
        if level >= 2:
            return self.no_food_limit_lv3
        elif level >= 1:
            return self.no_food_limit_lv2
        return self.no_food_limit_lv1


@dataclass(frozen=True)
class TrainConfig:
    episodes: int = 120000                  # 60K -> 120K

    alpha: float = 0.1
    gamma: float = 0.95

    # 三阶段 epsilon
    eps_start: float = 1.0
    eps_min: float = 0.02

    # Phase 1: ep 1~25000        快速探索  (eps: 1.0 -> ~0.22)
    eps_decay_phase1: float = 0.99994
    eps_phase2_start: int = 25000

    # Phase 2: ep 25001~70000    精炼学习  (eps: ~0.22 -> ~0.06)
    eps_decay_phase2: float = 0.99997
    eps_phase3_start: int = 70000

    # Phase 3: ep 70001~120000   巩固策略  (eps: ~0.06 -> ~0.02)
    eps_decay_phase3: float = 0.99998

    seed: int = 42

    eval_every: int = 2000
    eval_episodes: int = 50

    render_train: bool = False
    render_every: int = 10000
    fps_train: int = 30


@dataclass(frozen=True)
class UIConfig:
    cell_size: int = 20
    margin: int = 16
    panel_w: int = 320
    fps_demo: int = 15

    bg: Tuple = (20, 22, 30)
    grid_line: Tuple = (36, 40, 50)
    grid_bg: Tuple = (28, 30, 40)
    text: Tuple = (220, 225, 235)
    text_highlight: Tuple = (255, 220, 80)
    text_dim: Tuple = (120, 125, 135)

    snake_head: Tuple = (50, 215, 130)
    snake_body: Tuple = (35, 160, 100)
    snake_body_alt: Tuple = (30, 140, 90)
    food: Tuple = (240, 75, 75)
    bonus_food: Tuple = (255, 200, 50)
    bonus_food_glow: Tuple = (255, 230, 100)
    obstacle_static: Tuple = (100, 105, 130)
    obstacle_temp: Tuple = (180, 130, 200)
    obstacle_temp_flash: Tuple = (220, 180, 240)

    level_colors: Tuple = ((80, 200, 120), (200, 180, 60), (220, 80, 80))

    bar_bg: Tuple = (50, 54, 70)
    bar_fill_lv1: Tuple = (80, 200, 120)
    bar_fill_lv2: Tuple = (200, 180, 60)
    bar_fill_lv3: Tuple = (220, 80, 80)