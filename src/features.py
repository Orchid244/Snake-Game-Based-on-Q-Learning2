from __future__ import annotations
from typing import Tuple

UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3

DIR_VECS = {
    UP: (0, -1),
    RIGHT: (1, 0),
    DOWN: (0, 1),
    LEFT: (-1, 0),
}

ACT_LEFT, ACT_STRAIGHT, ACT_RIGHT = 0, 1, 2


def turn(dir_: int, rel_action: int) -> int:
    if rel_action == ACT_LEFT:
        return (dir_ - 1) % 4
    if rel_action == ACT_RIGHT:
        return (dir_ + 1) % 4
    return dir_


def next_pos(head_xy: Tuple[int, int], dir_: int) -> Tuple[int, int]:
    dx, dy = DIR_VECS[dir_]
    return (head_xy[0] + dx, head_xy[1] + dy)


def food_direction_bits(head_xy: Tuple[int, int], food_xy: Tuple[int, int]) -> Tuple[int, int, int, int]:
    hx, hy = head_xy
    fx, fy = food_xy
    return (
        1 if fx < hx else 0,
        1 if fx > hx else 0,
        1 if fy < hy else 0,
        1 if fy > hy else 0,
    )


def _get_level(env) -> int:
    if env.foods_eaten >= env.cfg.level3_threshold:
        return 2
    elif env.foods_eaten >= env.cfg.level2_threshold:
        return 1
    return 0


def _distance_category(env) -> int:
    """30x30: 0=close(<=5), 1=medium(<=12), 2=far(>12)"""
    hx, hy = env.snake[0]
    fx, fy = env.food
    dist = abs(hx - fx) + abs(hy - fy)
    if dist <= 5:
        return 0
    elif dist <= 12:
        return 1
    return 2


def _danger_2step(env, dir_abs: int) -> int:
    head = env.snake[0]
    pos1 = next_pos(head, dir_abs)
    if env.would_collide(pos1):
        return 1
    safe_count = 0
    for rel_a in [ACT_LEFT, ACT_STRAIGHT, ACT_RIGHT]:
        dir2 = turn(dir_abs, rel_a)
        pos2 = next_pos(pos1, dir2)
        if not env.would_collide(pos2):
            safe_count += 1
    return 1 if safe_count <= 1 else 0


def _escape_routes(env) -> int:
    hx, hy = env.snake[0]
    count = 0
    for d in [UP, RIGHT, DOWN, LEFT]:
        p = next_pos((hx, hy), d)
        if not env.would_collide(p):
            count += 1
    return min(count, 3)


def extract_state(env) -> tuple:
    """
    精简版 v5 状态（去掉 wall_prox 和 bonus_active）:
      (dangerL, dangerS, dangerR,        # 2^3 = 8
       danger2L, danger2S, danger2R,     # 2^3 = 8
       foodL, foodR, foodU, foodD,       # 2^4 = 16
       dir,                              # 4
       level,                            # 3
       escape_routes,                    # 4
       dist_cat)                         # 3

    Total: 8 * 8 * 16 * 4 * 3 * 4 * 3 = 147,456
    比 v4 的 884K 小 6 倍 → 学得更快！
    """
    head = env.snake[0]
    dir_ = env.dir

    dir_left = turn(dir_, ACT_LEFT)
    dir_straight = dir_
    dir_right = turn(dir_, ACT_RIGHT)

    dangerL = 1 if env.would_collide(next_pos(head, dir_left)) else 0
    dangerS = 1 if env.would_collide(next_pos(head, dir_straight)) else 0
    dangerR = 1 if env.would_collide(next_pos(head, dir_right)) else 0

    danger2L = _danger_2step(env, dir_left)
    danger2S = _danger_2step(env, dir_straight)
    danger2R = _danger_2step(env, dir_right)

    foodL, foodR, foodU, foodD = food_direction_bits(head, env.food)

    direction = dir_
    level = _get_level(env)
    escape = _escape_routes(env)
    dist_cat = _distance_category(env)

    return (dangerL, dangerS, dangerR,
            danger2L, danger2S, danger2R,
            foodL, foodR, foodU, foodD,
            direction,
            level,
            escape,
            dist_cat)