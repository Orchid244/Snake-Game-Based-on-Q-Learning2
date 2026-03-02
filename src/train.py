from __future__ import annotations
import os
import random
import time
import numpy as np

from config import GameConfig, TrainConfig, UIConfig
from env_snake import SnakeEnv
from qlearn import QLearner
from utils_io import ensure_dir, timestamp, save_json
from utils_plot import plot_learning_curves
from ui_pygame import SnakeUI


def evaluate(agent: QLearner, cfg: GameConfig, episodes: int, seed: int = 0):
    rng = random.Random(seed)
    wins, foods_list, ret_list, step_list, bonus_list = 0, [], [], [], []

    for _ in range(episodes):
        env = SnakeEnv(cfg, seed=rng.randrange(10 ** 9))
        s = env.reset()
        done = False
        G = 0.0
        info = {}
        while not done:
            a = agent.act(s, epsilon=0.0)
            s2, r, done, info = env.step(a)
            G += r
            s = s2
        wins += 1 if info.get("win", False) else 0
        foods_list.append(env.foods_eaten)
        ret_list.append(G)
        step_list.append(env.steps)
        bonus_list.append(env.bonus_eaten)

    n = max(1, episodes)
    return {
        "episodes": episodes,
        "win_rate": wins / n,
        "avg_foods": float(np.mean(foods_list)),
        "avg_return": float(np.mean(ret_list)),
        "avg_steps": float(np.mean(step_list)),
        "avg_bonus": float(np.mean(bonus_list)),
    }


def main():
    game_cfg = GameConfig()
    train_cfg = TrainConfig()
    ui_cfg = UIConfig()

    out_dir = os.path.join("results", f"run_{timestamp()}")
    ensure_dir(out_dir)

    print("=" * 65)
    print("  Snake Q-learning Training  v5  (30×30)")
    print(f"  Grid: {game_cfg.grid_w}×{game_cfg.grid_h}  Target: {game_cfg.target_foods} foods")
    print(f"  Levels: Lv2@{game_cfg.level2_threshold} Lv3@{game_cfg.level3_threshold}")
    print(f"  Episodes: {train_cfg.episodes}")
    print(f"  3-phase eps: {train_cfg.eps_decay_phase1} → "
          f"{train_cfg.eps_decay_phase2} → {train_cfg.eps_decay_phase3}")
    print(f"  Distance shaping: ±{game_cfg.reward_closer}")
    print(f"  State space: ~147K (sparse)")
    print(f"  Output: {out_dir}")
    print("=" * 65)

    env = SnakeEnv(game_cfg, seed=train_cfg.seed)
    agent = QLearner(n_actions=3, alpha=train_cfg.alpha, gamma=train_cfg.gamma, seed=train_cfg.seed)

    log = []
    epsilon = train_cfg.eps_start
    best_eval_wr = -1.0
    t_start = time.time()

    ui = None
    if train_cfg.render_train:
        ui = SnakeUI(ui_cfg, game_cfg.grid_w, game_cfg.grid_h, title="Training 30×30")

    for ep in range(1, train_cfg.episodes + 1):
        env.rng.seed(train_cfg.seed + ep)
        s = env.reset()
        done = False
        G = 0.0
        last_a = 1
        last_r = 0.0
        info = {}

        do_render = train_cfg.render_train and ui and (ep % max(1, train_cfg.render_every) == 0)

        while not done:
            a = agent.act(s, epsilon=epsilon)
            s2, r, done, info = env.step(a)
            agent.update(s, a, r, s2, done)
            s = s2
            G += r
            last_a = a
            last_r = r

            if do_render:
                evts = ui.handle_events()
                if evts["quit"]:
                    done = True
                    break
                ui.draw(env, hud={
                    "mode": "train",
                    "episode": ep,
                    "epsilon": epsilon,
                    "action": last_a,
                    "reward": last_r,
                    "return": G,
                    "step": env.steps,
                })
                ui.tick(train_cfg.fps_train)

        log.append({
            "episode": ep,
            "return": float(G),
            "foods": int(env.foods_eaten),
            "steps": int(env.steps),
            "win": bool(info.get("win", False)),
            "max_level": int(env.max_level),
            "bonus_eaten": int(env.bonus_eaten),
            "epsilon": float(epsilon),
        })

        # 三阶段 epsilon 衰减
        if ep < train_cfg.eps_phase2_start:
            epsilon = max(train_cfg.eps_min, epsilon * train_cfg.eps_decay_phase1)
        elif ep < train_cfg.eps_phase3_start:
            epsilon = max(train_cfg.eps_min, epsilon * train_cfg.eps_decay_phase2)
        else:
            epsilon = max(train_cfg.eps_min, epsilon * train_cfg.eps_decay_phase3)

        # Periodic eval
        if ep % max(1, train_cfg.eval_every) == 0:
            elapsed = time.time() - t_start
            eval_stats = evaluate(agent, game_cfg, episodes=train_cfg.eval_episodes, seed=train_cfg.seed + ep)
            tag = ""
            if eval_stats["win_rate"] > best_eval_wr:
                best_eval_wr = eval_stats["win_rate"]
                agent.save(os.path.join(out_dir, "q_table_best.pkl"))
                tag = " ** BEST **"

            phase = "P1" if ep < train_cfg.eps_phase2_start else ("P2" if ep < train_cfg.eps_phase3_start else "P3")
            print(f"[ep {ep:>6}] {elapsed:>7.0f}s {phase} eps={epsilon:.4f}  Q={agent.state_count:>6}  "
                  f"eval: wr={eval_stats['win_rate']:.1%} food={eval_stats['avg_foods']:.1f} "
                  f"ret={eval_stats['avg_return']:.1f} bonus={eval_stats['avg_bonus']:.1f}{tag}")
            save_json(eval_stats, os.path.join(out_dir, f"eval_ep{ep}.json"))

    # Final saves
    agent.save(os.path.join(out_dir, "q_table.pkl"))
    save_json(log, os.path.join(out_dir, "train_log.json"))
    plot_learning_curves(log, os.path.join(out_dir, "learning_curve.png"), window=1000)

    final = evaluate(agent, game_cfg, episodes=200, seed=99999)
    save_json(final, os.path.join(out_dir, "eval_final.json"))
    elapsed = time.time() - t_start
    print("\n" + "=" * 65)
    print(f"  FINAL EVALUATION (200 ep, eps=0)  [{elapsed:.0f}s total]")
    print(f"  Win rate:   {final['win_rate']:.1%}")
    print(f"  Avg foods:  {final['avg_foods']:.1f} / {game_cfg.target_foods}")
    print(f"  Avg return: {final['avg_return']:.1f}")
    print(f"  Avg bonus:  {final['avg_bonus']:.1f}")
    print("=" * 65)


if __name__ == "__main__":
    main()