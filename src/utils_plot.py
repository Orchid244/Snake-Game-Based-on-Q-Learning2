from typing import List, Dict
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils_io import ensure_dir


def _rolling_mean(x, window: int):
    if len(x) == 0:
        return np.array([])
    w = max(1, window)
    if len(x) < w:
        return np.array([np.mean(x)] * len(x))
    kernel = np.ones(w) / w
    return np.convolve(x, kernel, mode="valid")


def plot_learning_curves(log: List[Dict], out_path: str, window: int = 1000) -> None:
    ensure_dir(os.path.dirname(out_path) or ".")
    episodes = [r["episode"] for r in log]
    returns = [r["return"] for r in log]
    foods = [r["foods"] for r in log]
    wins = [1 if r["win"] else 0 for r in log]
    levels = [r.get("max_level", 0) for r in log]

    rm_ret = _rolling_mean(returns, window)
    rm_food = _rolling_mean(foods, window)
    rm_win = _rolling_mean(wins, window)
    rm_lvl = _rolling_mean(levels, window)

    fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)

    axes[0].plot(episodes, returns, alpha=0.08, color="steelblue")
    if len(rm_ret) > 0:
        axes[0].plot(episodes[len(episodes) - len(rm_ret):], rm_ret, linewidth=2.5, color="navy",
                     label=f"Rolling mean (w={window})")
    axes[0].set_ylabel("Return")
    axes[0].legend()
    axes[0].set_title("Training Curves (30×30, v5)")

    axes[1].plot(episodes, foods, alpha=0.08, color="green")
    if len(rm_food) > 0:
        axes[1].plot(episodes[len(episodes) - len(rm_food):], rm_food, linewidth=2.5, color="darkgreen",
                     label=f"Rolling mean (w={window})")
    axes[1].set_ylabel("Foods Eaten")
    axes[1].legend()

    axes[2].plot(episodes, wins, alpha=0.08, color="orange")
    if len(rm_win) > 0:
        axes[2].plot(episodes[len(episodes) - len(rm_win):], rm_win, linewidth=2.5, color="darkorange",
                     label=f"Rolling mean (w={window})")
    axes[2].set_ylabel("Win Rate")
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].legend()

    axes[3].plot(episodes, levels, alpha=0.08, color="purple")
    if len(rm_lvl) > 0:
        axes[3].plot(episodes[len(episodes) - len(rm_lvl):], rm_lvl, linewidth=2.5, color="darkviolet",
                     label=f"Rolling mean (w={window})")
    axes[3].set_ylabel("Max Level Reached")
    axes[3].set_xlabel("Episode")
    axes[3].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"  Learning curve saved: {out_path}")