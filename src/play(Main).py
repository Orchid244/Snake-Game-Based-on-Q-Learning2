from __future__ import annotations
import os
import sys
import glob
import pygame

from config import GameConfig, UIConfig
from env_snake import SnakeEnv
from qlearn import QLearner
from ui_pygame import SnakeUI


def _find_qtable(results_dir: str = "results") -> str:
    runs = sorted(glob.glob(os.path.join(results_dir, "run_*")))
    if not runs:
        print("ERROR: No results/run_* found. Run training first: python -m src.train")
        sys.exit(1)
    latest = runs[-1]
    best = os.path.join(latest, "q_table_best.pkl")
    if os.path.exists(best):
        return best
    final = os.path.join(latest, "q_table.pkl")
    if os.path.exists(final):
        return final
    print(f"ERROR: No q_table found in {latest}")
    sys.exit(1)


def main():
    game_cfg = GameConfig()
    ui_cfg = UIConfig()

    q_path = _find_qtable()
    print(f"Loading Q-table from: {q_path}")
    agent = QLearner.load(q_path)
    print(f"  Q-table states: {agent.state_count}")

    env = SnakeEnv(game_cfg, seed=42)
    s = env.reset()

    ui = SnakeUI(ui_cfg, game_cfg.grid_w, game_cfg.grid_h, title="Snake 30×30 Demo")
    episode = 1
    G = 0.0
    last_r = 0.0
    last_a = 1
    wins = 0
    total = 0
    paused = False

    running = True
    while running:
        evts = ui.handle_events()
        if evts["quit"]:
            running = False
            continue
        if evts["reset"]:
            env.rng.seed(episode * 7 + 13)
            s = env.reset()
            episode += 1
            G = 0.0
            last_r = 0.0
            last_a = 1
            continue
        if evts["pause"]:
            paused = not paused

        if paused:
            ui.draw(env, hud={
                "mode": "PAUSED",
                "episode": episode,
                "epsilon": 0.0,
                "action": last_a,
                "reward": last_r,
                "return": G,
                "step": env.steps,
            })
            ui.tick(10)
            continue

        a = agent.act(s, epsilon=0.0)
        s2, r, done, info = env.step(a)
        s = s2
        last_a = a
        last_r = r
        G += r

        ui.draw(env, hud={
            "mode": f"demo  W:{wins}/{total}",
            "episode": episode,
            "epsilon": 0.0,
            "action": last_a,
            "reward": last_r,
            "return": G,
            "step": env.steps,
        })
        ui.tick(ui_cfg.fps_demo)

        if done:
            total += 1
            if info.get("win", False):
                wins += 1
            pygame.time.wait(600)
            env.rng.seed(episode * 7 + 13)
            s = env.reset()
            episode += 1
            G = 0.0
            last_r = 0.0
            last_a = 1

    pygame.quit()
    print(f"\nDemo ended. Wins: {wins}/{total}")


if __name__ == "__main__":
    main()