from __future__ import annotations
from typing import Dict
import pygame
import math

from config import UIConfig
from features import ACT_LEFT, ACT_STRAIGHT, ACT_RIGHT

_ACT_NAMES = {ACT_LEFT: "Left", ACT_STRAIGHT: "Straight", ACT_RIGHT: "Right"}
_DIR_NAMES = ["Up", "Right", "Down", "Left"]
_LEVEL_NAMES = ["Level 1", "Level 2", "Level 3"]


class SnakeUI:
    def __init__(self, ui_cfg: UIConfig, grid_w: int, grid_h: int, title: str = "Snake Q-learning"):
        pygame.init()
        pygame.display.set_caption(title)
        self.ui = ui_cfg
        self.gw = grid_w
        self.gh = grid_h
        cs = ui_cfg.cell_size
        m = ui_cfg.margin

        self.grid_px_w = grid_w * cs
        self.grid_px_h = grid_h * cs
        self.win_w = m * 2 + self.grid_px_w + ui_cfg.panel_w
        self.win_h = m * 2 + self.grid_px_h

        self.screen = pygame.display.set_mode((self.win_w, self.win_h))
        self.clock = pygame.time.Clock()

        self.font_title = pygame.font.SysFont("consolas", 22, bold=True)
        self.font = pygame.font.SysFont("consolas", 16)
        self.font_small = pygame.font.SysFont("consolas", 13)

    def tick(self, fps: int) -> None:
        self.clock.tick(fps)

    def handle_events(self):
        events = {"quit": False, "reset": False, "pause": False}
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                events["quit"] = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    events["quit"] = True
                if event.key == pygame.K_r:
                    events["reset"] = True
                if event.key == pygame.K_SPACE:
                    events["pause"] = True
        return events

    def draw(self, env, hud: Dict) -> None:
        self.screen.fill(self.ui.bg)
        self._draw_grid_bg()
        self._draw_entities(env)
        self._draw_panel(env, hud)
        pygame.display.flip()

    def _draw_grid_bg(self) -> None:
        m = self.ui.margin
        cs = self.ui.cell_size
        grid_rect = pygame.Rect(m, m, self.grid_px_w, self.grid_px_h)
        pygame.draw.rect(self.screen, self.ui.grid_bg, grid_rect)
        for x in range(self.gw + 1):
            px = m + x * cs
            w = 2 if x % 5 == 0 else 1
            color = self.ui.text_dim if x % 5 == 0 else self.ui.grid_line
            pygame.draw.line(self.screen, color, (px, m), (px, m + self.grid_px_h), width=w)
        for y in range(self.gh + 1):
            py = m + y * cs
            w = 2 if y % 5 == 0 else 1
            color = self.ui.text_dim if y % 5 == 0 else self.ui.grid_line
            pygame.draw.line(self.screen, color, (m, py), (m + self.grid_px_w, py), width=w)

    def _cell_rect(self, x: int, y: int, pad: int = 1) -> pygame.Rect:
        m = self.ui.margin
        cs = self.ui.cell_size
        return pygame.Rect(m + x * cs + pad, m + y * cs + pad, cs - pad * 2, cs - pad * 2)

    def _draw_entities(self, env) -> None:
        step = env.steps

        for pos in env.static_obstacles:
            pygame.draw.rect(self.screen, self.ui.obstacle_static, self._cell_rect(*pos))

        for pos, ttl in env.temp_obstacles.items():
            if ttl <= 10 and step % 4 < 2:
                color = self.ui.obstacle_temp_flash
            else:
                color = self.ui.obstacle_temp
            pygame.draw.rect(self.screen, color, self._cell_rect(*pos))

        fx, fy = env.food
        r = self._cell_rect(fx, fy, pad=2)
        pygame.draw.rect(self.screen, self.ui.food, r, border_radius=4)

        if env.bonus_food is not None:
            bx, by = env.bonus_food
            pulse = abs(math.sin(step * 0.3)) * 0.4 + 0.6
            br = self._cell_rect(bx, by, pad=1)
            glow_r = br.inflate(3, 3)
            glow_color = tuple(int(c * pulse) for c in self.ui.bonus_food_glow)
            pygame.draw.rect(self.screen, glow_color, glow_r, border_radius=5)
            pygame.draw.rect(self.screen, self.ui.bonus_food, br, border_radius=4)

        for i, (x, y) in enumerate(env.snake):
            if i == 0:
                pygame.draw.rect(self.screen, self.ui.snake_head, self._cell_rect(x, y, pad=0), border_radius=4)
            else:
                color = self.ui.snake_body if i % 2 == 0 else self.ui.snake_body_alt
                pygame.draw.rect(self.screen, color, self._cell_rect(x, y, pad=1), border_radius=2)

    def _draw_panel(self, env, hud: Dict) -> None:
        m = self.ui.margin
        x0 = m + self.grid_px_w + 20
        y = m

        title = self.font_title.render("SNAKE 30x30", True, self.ui.text)
        self.screen.blit(title, (x0, y))
        y += 26
        mode_str = hud.get("mode", "demo").upper()
        mode_surf = self.font.render(f"[{mode_str}]", True, self.ui.text_dim)
        self.screen.blit(mode_surf, (x0, y))
        y += 24

        level = env.current_level
        lv_color = self.ui.level_colors[min(level, 2)]
        lv_surf = self.font.render(f"  {_LEVEL_NAMES[min(level, 2)]}", True, lv_color)
        self.screen.blit(lv_surf, (x0, y))
        y += 24

        bar_w = self.ui.panel_w - 40
        bar_h = 14
        bar_rect = pygame.Rect(x0, y, bar_w, bar_h)
        pygame.draw.rect(self.screen, self.ui.bar_bg, bar_rect, border_radius=4)
        fill_frac = min(1.0, env.foods_eaten / max(1, env.cfg.target_foods))
        fill_rect = pygame.Rect(x0, y, int(bar_w * fill_frac), bar_h)
        bar_colors = [self.ui.bar_fill_lv1, self.ui.bar_fill_lv2, self.ui.bar_fill_lv3]
        pygame.draw.rect(self.screen, bar_colors[min(level, 2)], fill_rect, border_radius=4)
        pct = self.font_small.render(f"{env.foods_eaten}/{env.cfg.target_foods}", True, self.ui.text)
        self.screen.blit(pct, (x0 + bar_w + 4, y))
        y += 22

        pygame.draw.line(self.screen, self.ui.grid_line, (x0, y), (x0 + self.ui.panel_w - 40, y))
        y += 8

        lines = [
            ("Episode", str(hud.get("episode", "-")), self.ui.text),
            ("Step", str(hud.get("step", env.steps)), self.ui.text),
            ("Epsilon", f'{hud.get("epsilon", 0):.4f}', self.ui.text),
            ("", "", self.ui.text),
            ("Action", _ACT_NAMES.get(hud.get("action", 1), "?"), self.ui.text_highlight),
            ("Direction", _DIR_NAMES[env.dir % 4], self.ui.text),
            ("Reward", f'{hud.get("reward", 0):.3f}', self.ui.text_highlight),
            ("Return", f'{hud.get("return", 0):.2f}', self.ui.text),
            ("", "", self.ui.text),
            ("Length", str(len(env.snake)), self.ui.text),
            ("Bonus Eaten", str(env.bonus_eaten), self.ui.bonus_food),
            ("Static Obst", str(len(env.static_obstacles)), self.ui.obstacle_static),
            ("Temp Obst", str(len(env.temp_obstacles)), self.ui.obstacle_temp),
            ("NoFood Steps", str(env.no_food_steps), self.ui.text_dim),
        ]

        if env.bonus_food is not None:
            lines.append(("Bonus Timer", str(env.bonus_food_timer), self.ui.bonus_food))

        for label, val, color in lines:
            if label == "" and val == "":
                y += 4
                continue
            s = f"{label:>13}: {val}"
            surf = self.font.render(s, True, color)
            self.screen.blit(surf, (x0, y))
            y += 20

        y = self.win_h - m - 60
        pygame.draw.line(self.screen, self.ui.grid_line, (x0, y), (x0 + self.ui.panel_w - 40, y))
        y += 6
        for hint in ["ESC: Quit", "R: Reset", "SPACE: Pause"]:
            surf = self.font_small.render(hint, True, self.ui.text_dim)
            self.screen.blit(surf, (x0, y))
            y += 16
