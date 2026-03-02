
# Snake Game Based on Q-Learning

YouTube: *(to be added)*

GitHub: *(to be added)*

## 1 Introduction

This project uses the Q-Learning algorithm to design and implement a Snake game. In this game, an AI agent controls the snake to navigate a 30×30 grid, eat food, avoid obstacles, and survive as long as possible. The agent makes decisions based on the current state of the environment using a learned Q-table. This project uses the following technology stack. Python is used as the main programming language. The Q-Learning algorithm is implemented with a tabular approach using a compact, hand-crafted feature-based state representation to keep learning feasible on the large grid. Pygame is a library for game development that provides functions such as graphics, input processing, and UI rendering. We use this library to implement game operation, dynamic updates, collision detection, and real-time visualization of the agent's decisions.

## 2 Game Design

### 2.1 Rules of the Game

In this game, the snake starts near the center of a 30×30 grid. At each time step, the agent selects a movement action and the snake moves one cell in the corresponding direction. The main objective is to eat food items to increase the score and length while avoiding death. The episode ends if any of the following conditions is met: the snake collides with the wall (out of bounds), the snake collides with its own body, the snake collides with an obstacle, the snake fails to reach food within a step limit (no-food timeout), or a maximum step cap is reached.

To increase the challenge progressively, the environment introduces three difficulty levels based on the number of foods eaten:

- **Level 1:** Only normal food (red) exists on the grid. The snake simply needs to find and eat food while avoiding walls and its own body.
- **Level 2:** Static obstacles (gray/blue) start spawning on the grid each time food is eaten. Bonus food (gold/yellow) may also appear with a countdown timer, offering higher reward but requiring the agent to reach it before it expires.
- **Level 3:** In addition to static obstacles, temporary obstacles (purple) begin spawning. These obstacles disappear after a fixed number of steps, and they flash when they are about to expire, providing a visual cue to the agent and the viewer.

Normal food increases the snake's length by 1 and provides a standard reward. Bonus food gives a higher reward and is time-limited, encouraging the agent to balance safety and opportunity.

### 2.2 Class Design of the Game

**SnakeEnv class** responsible for managing the game environment and state. It handles the grid rules, snake movement, collision detection, food spawning, obstacle spawning (both static and temporary), timers, level transitions, and reward calculation.

**Features module** used to extract a compact discrete state from the environment. It converts the full game state into a tuple of hand-crafted features including danger detection, food direction, distance category, escape routes, and current level.

**QLearner class** represents the Q-learning agent. It stores Q-values in a dictionary-based Q-table and implements epsilon-greedy action selection, Q-value updates, and save/load functionality.

**Training script (train.py)** manages the training loop. It runs episodes, updates the Q-table, performs periodic evaluation with greedy policy, logs results, and saves training artifacts including Q-tables and learning curves.

**Play script (play(Main).py)** loads a trained Q-table and runs the game in demo mode with the Pygame UI, allowing real-time observation of the learned policy.

**SnakeUI class (ui_pygame.py)** handles all Pygame rendering. It draws the grid and all game entities with distinct colors, and displays a side panel (HUD) showing real-time information such as level, episode, step, action, direction, reward, return, snake length, bonus eaten count, and obstacle counts.

**Config module (config.py)** stores all configurable parameters including grid size, level thresholds, obstacle settings, reward values, training hyperparameters, and UI rendering options.

**Utility modules (utils_io.py, utils_plot.py)** provide helper functions for file I/O (saving/loading Q-tables and JSON logs) and for generating training curve plots.

### 2.3 UI Design

The user interface follows a clean and informative style. The main game area renders the 30×30 grid with colored cells representing different entities. The right-hand side of the screen features a dark panel displaying real-time game statistics and agent information. The color scheme is as follows:

- **Green (light/dark alternating):** Snake head and body
- **Red:** Normal food
- **Gold/Yellow (with glow effect):** Bonus food (timed)
- **Gray/Blue:** Static obstacles (persistent)
- **Purple:** Temporary obstacles (expire after a lifetime)
- **Flashing Purple:** Temporary obstacles about to expire (remaining steps ≤ 10)

The HUD panel displays: Level, Episode, Step, Action, Direction, Reward, Return, Length, Bonus Eaten, Static Obs count, Temp Obs count, and Refuel Steps.

*(Insert screenshot: game UI overview here)*

*(Insert screenshot: Level 1 vs Level 3 comparison here)*

## 3 Implement of Q-learning Algorithm

Q-Learning is a value-based reinforcement learning algorithm that selects the best action by learning a Q-value function. In this project, since the state space is large (30×30 grid with many possible configurations), using the raw board as state is infeasible for tabular Q-learning. Therefore, we designed a compact feature-based state representation that captures the most decision-relevant information while keeping the Q-table size manageable.

**Action Space:** Instead of using absolute directions (Up/Down/Left/Right), the agent uses three relative actions based on its current heading: Turn Left, Go Straight, and Turn Right. This design eliminates directional symmetry and simplifies the learning problem.

**State Representation:** The state is encoded as a discrete tuple consisting of the following features:

- Immediate danger on left, straight, and right (whether the next cell in each relative direction would cause a collision)
- Two-step dead-end danger on left, straight, and right (whether moving in that direction leads to a position with very few safe exits)
- Food direction bits indicating whether the food is to the left, right, above, or below the snake's head
- Current movement direction (4 possibilities)
- Current difficulty level (1, 2, or 3)
- Number of safe escape routes from the head position (clipped to a maximum value)
- Distance category to food based on Manhattan distance (close, medium, or far)

**Reward Design:** The reward function is designed to guide learning while preventing degenerate strategies. It includes: a small step penalty to encourage efficiency, a positive reward for eating normal food, a larger positive reward for eating bonus food, a negative reward for collision or death, a negative reward for no-food timeout, a win bonus for reaching the target number of foods, and distance-based reward shaping that gives positive feedback when the snake moves closer to food and negative feedback when it moves farther away. The distance shaping is particularly important for the 30×30 grid, as random exploration alone would be extremely inefficient.

**Q-learning Update:** The Q-table is updated using the standard Q-learning rule:

Q(s, a) ← (1 - α) · Q(s, a) + α · [r + γ · max_a' Q(s', a')]

where α is the learning rate, γ is the discount factor, r is the immediate reward, and s' is the next state. Action selection during training uses the ε-greedy strategy: with probability ε the agent selects a random action (exploration), and otherwise it selects the action with the highest Q-value (exploitation). The value of ε is gradually decayed over training using a multi-phase schedule to shift from exploration to exploitation.

**Training Setup:** Training runs for a large number of episodes. Periodic evaluation is performed at fixed intervals using a greedy policy (ε = 0) to track the true performance of the learned policy. All training logs, Q-tables, and evaluation results are saved under the results directory.

*(Insert screenshot: training curves here)*

## 4 Challenges and Solutions

**Challenge 1: Large grid makes learning sparse.** In a 30×30 grid, food can be far away from the snake. Without guidance, the agent wanders for many steps receiving only step penalties, making learning unstable and slow. To address this, we introduced distance-based reward shaping that provides directional feedback (positive for moving closer to food, negative for moving farther). Combined with a distance category feature in the state, this significantly improved learning speed and stability on the large map.

**Challenge 2: State space explosion.** Tabular Q-learning becomes infeasible if the state is too detailed, as the Q-table becomes extremely sparse and most entries are never visited. To solve this, we designed a feature-based discrete state that captures only the most decision-relevant information (local danger, food direction, escape routes, distance bins, and level). This keeps the Q-table compact enough for effective learning while still encoding survival-critical signals.

**Challenge 3: Increasing difficulty causes performance collapse.** Adding obstacles too aggressively when the snake reaches higher levels can cause the learned policy to fail suddenly, as the agent encounters situations it has not been trained on. We addressed this by carefully balancing the difficulty progression: controlling obstacle spawning frequency, setting appropriate obstacle trigger thresholds, and giving temporary obstacles a finite lifetime so that the map does not become permanently blocked.

**Challenge 4: Avoiding local traps and dead-ends.** Many deaths in Snake occur not from obvious collisions but from the snake entering narrow corridors that become dead-ends as its body grows. To mitigate this, we added a two-step dead-end danger check and an escape route count feature to the state representation. These features bias the agent away from potentially fatal positions even when the immediate next step appears safe.

## 5 Conclusion

During training, we observed that the performance of the agent gradually improved as training progressed. The snake learned how to efficiently navigate toward food, avoid walls and obstacles, and survive through all three difficulty levels. The training curves showed consistent improvement in return, foods eaten, and win rate over the course of training. The agent achieved a good balance between exploration and exploitation, and was able to maintain stable performance while adapting to increasingly difficult environments.

Through this project, we successfully implemented a Snake game based on the Q-Learning algorithm on a challenging 30×30 grid with multi-level difficulty. The agent can continuously optimize its strategy through the learning algorithm and improve its survival and food-collection ability. This project demonstrates the application of tabular Q-learning in a game environment with progressive complexity, and provides an interesting experimental platform for exploring both the potential and the limitations of reinforcement learning. Although tabular Q-learning relies on hand-crafted features and has limited long-horizon planning capability, the results show that with careful state design and reward shaping, it can achieve strong performance even in a large and dynamic environment. Future work could explore deep Q-network (DQN) approaches using raw grid observations as input, or more advanced exploration strategies for even larger or more complex maps.

## 6 Project Structure

```
Assignment1_V4/
  src/
    config.py          — Game, training, and UI parameters
    env_snake.py        — Snake environment (rules, rewards, obstacles, timers)
    features.py         — State feature extraction
    qlearn.py           — Q-table agent (act, update, save, load)
    train.py            — Training loop, evaluation, logging, plots
    play(Main).py       — Load Q-table and run demo with Pygame UI
    ui_pygame.py        — Pygame rendering and HUD panel
    utils_io.py         — File I/O utilities (save/load Q-tables and logs)
    utils_plot.py       — Training curve plotting utilities
    requirements.txt    — Python dependencies
    readme.md           — This file
    results/
      run_YYYYMMDD_HHMMSS/
        q_table.pkl
        q_table_best.pkl
        train_log.json
        eval_epXXXX.json
        eval_final.json
        learning_curve.png
```

## 7 Reference

[1] Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.

[2] Watkins, C. J. C. H. (1989). *Learning from Delayed Rewards* (PhD thesis). University of Cambridge.

[3] Pygame Documentation. https://www.pygame.org/docs/
