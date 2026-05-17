import math
import random
import sys
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

import csv
import json
import os
from datetime import datetime
from dataclasses import asdict

# ============================================================
# Config
# ============================================================

GRID_WIDTH = 30
GRID_HEIGHT = 30
CELL_SIZE = 32
PANEL_MARGIN = 20
SENSOR_CELL_SIZE = 40
SENSOR_GRID_SIZE = 3 * SENSOR_CELL_SIZE
DEBUG_PANEL_WIDTH = 260


SMELL_BAR_MAX_HEIGHT = 80
SMELL_BAR_WIDTH = 10
SMELL_BAR_GAP = 4
SMELL_EPSILON = 1e-6
FOOD1_MASS = 1.0
FOOD2_MASS = 1.0

WINDOW_WIDTH = GRID_WIDTH * CELL_SIZE + DEBUG_PANEL_WIDTH
WINDOW_HEIGHT = max(GRID_HEIGHT * CELL_SIZE, 520)
FPS = 12

EMPTY = 0
AGENT = 1
FOOD1 = 2
FOOD2 = 3
WALL = 4

# ============================================================
# Continual-learning methodology config
# ============================================================

GLOBAL_ENERGY_E = 20.0
NUM_TRAIN_WORLDS = 80
TRAIN_WORLD_SEED = 123



COLORS = {
    EMPTY: (30, 30, 30),
    AGENT: (255, 255, 255),
    FOOD1: (0, 255, 0),
    FOOD2: (255, 0, 0),
    WALL: (100, 100, 100),
    "GRID": (70, 70, 70),
    "BG": (15, 15, 15),
}


@dataclass
class WorldSpec:
    width: int
    height: int
    agent_start: Tuple[int, int]
    food1_positions: List[Tuple[int, int]]
    food2_positions: List[Tuple[int, int]]
    food1_energy: float
    food2_energy: float
    epsilon1: float = 1.0
    epsilon2: float = 1.0
    world_id: int = -1
    randomize_agent_start: bool = False
    action_permutation: Optional[List[int]] = None
    observation_permutation: Optional[List[int]] = None

@dataclass
class EnvConfig:
    width: int = 40
    height: int = 40
    num_food1: int = 30
    num_food2: int = 30
    food1_energy: int = 20
    food2_energy: int = 20
    episode_horizon: int = 100

    # survival
    use_survival: bool = True
    initial_energy: float = 40.0
    step_energy_cost: float = 1.0
    max_energy: float = 80.0
    end_episode_on_zero_energy: bool = True




@dataclass
class NetworkConfig:
    hidden_sizes: List[int] = field(default_factory=lambda: [128, 128])
    activation: str = "tanh"   # "tanh" or "relu"


@dataclass
class PPOConfig:
    total_env_steps_target: int = 1_000_000
    steps_per_update: int = 2048
    ppo_epochs: int = 10
    minibatch_size: int = 256

    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    learning_rate: float = 3e-4
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5

    steps_per_world: int = 20_000

    use_l2_regularization: bool = False
    l2_coefficient: float = 1e-4

    use_continual_backprop: bool = False
    cbp_decay: float = 0.99
    cbp_reinit_fraction: float = 1e-4
    cbp_min_steps_before_reinit: int = 0

    # NEW: activation/plasticity diagnostics
    activation_log_every_updates: int = 10
    activation_sample_size: int = 2048
    dormant_threshold: float = 0.01

    device: str = "cpu"
    seed: int = 1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999

# ============================================================
# Environment
# ============================================================
def generate_world_spec(
    cfg: EnvConfig,
    rng: random.Random,
    food1_energy: float,
    food2_energy: float,
    epsilon1: float = 1.0,
    epsilon2: float = 1.0,
    world_id: int = -1,
) -> WorldSpec:
    occupied = set()

    def random_empty():
        while True:
            x = rng.randrange(cfg.width)
            y = rng.randrange(cfg.height)
            if (x, y) not in occupied:
                occupied.add((x, y))
                return (x, y)

    agent_start = random_empty()
    food1_positions = [random_empty() for _ in range(cfg.num_food1)]
    food2_positions = [random_empty() for _ in range(cfg.num_food2)]

    return WorldSpec(
        width=cfg.width,
        height=cfg.height,
        agent_start=agent_start,
        food1_positions=food1_positions,
        food2_positions=food2_positions,
        food1_energy=float(food1_energy),
        food2_energy=float(food2_energy),
        epsilon1=float(epsilon1),
        epsilon2=float(epsilon2),
        world_id=world_id,
    )

def generate_base_world(cfg: EnvConfig, seed: int) -> WorldSpec:
    rng = random.Random(seed)
    return generate_world_spec(
        cfg=cfg,
        rng=rng,
        food1_energy=0.0,   # placeholder, will be overwritten per phase
        food2_energy=0.0,
        epsilon1=1.0,
        epsilon2=1.0,
        world_id=0,
    )


def generate_training_worlds(
    cfg: EnvConfig,
    num_worlds: int,
    E: float,
    seed: int,
) -> List[WorldSpec]:
    rng = random.Random(seed)
    worlds = []

    for i in range(num_worlds):
        small_coeff = rng.uniform(-0.1, 0.1)

        if i % 2 == 0:
            food1_energy = E
            food2_energy = small_coeff * E
        else:
            food1_energy = small_coeff * E
            food2_energy = E

        spec = generate_world_spec(
            cfg=cfg,
            rng=rng,
            food1_energy=food1_energy,
            food2_energy=food2_energy,
            world_id=i,
        )
        spec.randomize_agent_start = True

        perm = [0, 1, 2, 3]
        rng.shuffle(perm)
        spec.action_permutation = perm

        obs_dim = 3 * 3 * 5 + 8
        obs_perm = list(range(obs_dim))
        rng.shuffle(obs_perm)
        spec.observation_permutation = obs_perm

        worlds.append(spec)

    return worlds


class GridWorld:

    ACTIONS = [
        (0, -1),   # up
        (0, 1),    # down
        (-1, 0),   # left
        (1, 0),    # right
    ]

    def __init__(self, cfg: EnvConfig, seed: int = 0):
        self.cfg = cfg
        self.width = cfg.width
        self.height = cfg.height
        self.num_food1 = cfg.num_food1
        self.num_food2 = cfg.num_food2
        self.food1_energy = cfg.food1_energy
        self.food2_energy = cfg.food2_energy
        self.episode_horizon = cfg.episode_horizon

        self.grid = [[EMPTY for _ in range(self.width)] for _ in range(self.height)]
        self.agent_pos = (0, 0)
        self.food_eaten = 0
        self.episode_return = 0.0
        self.corner_smell_food1 = [0.0, 0.0, 0.0, 0.0]
        self.corner_smell_food2 = [0.0, 0.0, 0.0, 0.0]
        self.smell_max_value = 1.0

        self.use_survival = cfg.use_survival
        self.initial_energy = float(cfg.initial_energy)
        self.step_energy_cost = float(cfg.step_energy_cost)
        self.max_energy = float(cfg.max_energy)
        self.end_episode_on_zero_energy = bool(cfg.end_episode_on_zero_energy)

        self.energy = self.initial_energy
        self.energy_sum = 0.0

        self.rng = random.Random(seed)
        self.action_permutation = [0, 1, 2, 3]
        self.observation_permutation = list(range(self.obs_size))
        self.reset()

    def _random_empty_cell(self):
        while True:
            x = self.rng.randrange(self.width)
            y = self.rng.randrange(self.height)
            if self.grid[y][x] == EMPTY:
                return x, y

    def _respawn_food(self, food_tile: int):
        x, y = self._random_empty_cell()
        self.grid[y][x] = food_tile

    def load_world(self, world_spec: WorldSpec):
        if world_spec.action_permutation is None:
            self.action_permutation = [0, 1, 2, 3]
        else:
            self.action_permutation = list(world_spec.action_permutation)

        if world_spec.observation_permutation is None:
            self.observation_permutation = list(range(self.obs_size))
        else:
            self.observation_permutation = list(world_spec.observation_permutation)

        self.width = world_spec.width
        self.height = world_spec.height
        self.food1_energy = float(world_spec.food1_energy)
        self.food2_energy = float(world_spec.food2_energy)

        self.grid = [[EMPTY for _ in range(self.width)] for _ in range(self.height)]
        self.food_eaten = 0
        self.steps_alive = 0
        self.episode_return = 0.0
        self.energy = self.initial_energy
        self.energy_sum = self.energy

        for x, y in world_spec.food1_positions:
            self.grid[y][x] = FOOD1

        for x, y in world_spec.food2_positions:
            self.grid[y][x] = FOOD2

        if world_spec.randomize_agent_start:
            occupied = set(world_spec.food1_positions) | set(world_spec.food2_positions)
            while True:
                ax = self.rng.randrange(self.width)
                ay = self.rng.randrange(self.height)
                if (ax, ay) not in occupied:
                    break
        else:
            ax, ay = world_spec.agent_start

        self.agent_pos = (ax, ay)
        self.grid[ay][ax] = AGENT

        self.recompute_corner_smells()
        return self.get_observation()

    def draw_smell_panel(self, screen, font, panel_x, panel_y):
        """
        Draw a smell visualization panel below the 3x3 vision panel.
        Shows 4 bar pairs corresponding to the world corners:
          TL, TR, BL, BR
        Green = FOOD1 smell
        Red   = FOOD2 smell
        """
        title = font.render("Corner Smell", True, (255, 255, 255))
        screen.blit(title, (panel_x, panel_y))

        labels = ["TL", "TR", "BL", "BR"]

        chart_top = panel_y + 35
        chart_height = 100
        baseline_y = chart_top + chart_height
        group_spacing = 52
        start_x = panel_x + 8

        for i, label_text in enumerate(labels):
            smell1 = self.corner_smell_food1[i]
            smell2 = self.corner_smell_food2[i]

            h1 = int((smell1 / self.smell_max_value) * SMELL_BAR_MAX_HEIGHT)
            h2 = int((smell2 / self.smell_max_value) * SMELL_BAR_MAX_HEIGHT)

            gx = start_x + i * group_spacing

            rect1 = pygame.Rect(gx, baseline_y - h1, SMELL_BAR_WIDTH, h1)
            rect2 = pygame.Rect(gx + SMELL_BAR_WIDTH + SMELL_BAR_GAP, baseline_y - h2, SMELL_BAR_WIDTH, h2)

            pygame.draw.rect(screen, COLORS[FOOD1], rect1)
            pygame.draw.rect(screen, COLORS[FOOD2], rect2)

            pygame.draw.rect(screen, COLORS["GRID"], rect1, 1)
            pygame.draw.rect(screen, COLORS["GRID"], rect2, 1)

            # corner label below the pair
            label = font.render(label_text, True, (255, 255, 255))
            screen.blit(label, (gx - 2, baseline_y + 6))

        # baseline
        pygame.draw.line(
            screen,
            COLORS["GRID"],
            (start_x - 4, baseline_y),
            (start_x + 3 * group_spacing + 30, baseline_y),
            2
        )

        # legend
        legend_y = baseline_y + 35

        sw1 = pygame.Rect(panel_x, legend_y, 18, 18)
        pygame.draw.rect(screen, COLORS[FOOD1], sw1)
        pygame.draw.rect(screen, COLORS["GRID"], sw1, 1)
        txt1 = font.render("FOOD1", True, (255, 255, 255))
        screen.blit(txt1, (panel_x + 26, legend_y - 2))

        sw2 = pygame.Rect(panel_x, legend_y + 24, 18, 18)
        pygame.draw.rect(screen, COLORS[FOOD2], sw2)
        pygame.draw.rect(screen, COLORS["GRID"], sw2, 1)
        txt2 = font.render("FOOD2", True, (255, 255, 255))
        screen.blit(txt2, (panel_x + 26, legend_y + 22))

    def get_corner_positions(self):
        """
        Returns the 4 corners of the agent's current cell in continuous grid coordinates.
        Order:
          0 = top-left
          1 = top-right
          2 = bottom-left
          3 = bottom-right
        """
        ax, ay = self.agent_pos
        return [
            (float(ax), float(ay)),
            (float(ax + 1), float(ay)),
            (float(ax), float(ay + 1)),
            (float(ax + 1), float(ay + 1)),
        ]

    def get_cell_center(self, x: int, y: int):
        """
        Treat each tile as occupying [x, x+1] x [y, y+1], so its center is (x+0.5, y+0.5).
        """
        return (x + 0.5, y + 0.5)

    def compute_smell_for_food_type(self, target_tile: int, mass: float):
        """
        Smell at each corner q:
            sum( mass(cell,target_tile) / dist(cell_center, q)^2 )
        """
        corners = self.get_corner_positions()
        smells = [0.0, 0.0, 0.0, 0.0]

        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y][x] != target_tile:
                    continue

                cx, cy = self.get_cell_center(x, y)

                for i, (qx, qy) in enumerate(corners):
                    dx = cx - qx
                    dy = cy - qy
                    dist_sq = dx * dx + dy * dy + SMELL_EPSILON
                    smells[i] += mass / dist_sq

        return smells

    def recompute_corner_smells(self):
        self.corner_smell_food1 = self.compute_smell_for_food_type(FOOD1, FOOD1_MASS)
        self.corner_smell_food2 = self.compute_smell_for_food_type(FOOD2, FOOD2_MASS)

        max_smell = max(
            max(self.corner_smell_food1, default=0.0),
            max(self.corner_smell_food2, default=0.0),
            1.0,
        )
        self.smell_max_value = max_smell

    def reset(self, world_spec: Optional[WorldSpec] = None):
        if world_spec is not None:
            return self.load_world(world_spec)

        self.grid = [[EMPTY for _ in range(self.width)] for _ in range(self.height)]
        self.food_eaten = 0
        self.steps_alive = 0
        self.episode_return = 0.0
        self.energy = self.initial_energy
        self.energy_sum = self.energy

        self.agent_pos = self._random_empty_cell()
        ax, ay = self.agent_pos
        self.grid[ay][ax] = AGENT

        for _ in range(self.num_food1):
            x, y = self._random_empty_cell()
            self.grid[y][x] = FOOD1

        for _ in range(self.num_food2):
            x, y = self._random_empty_cell()
            self.grid[y][x] = FOOD2

        self.recompute_corner_smells()
        return self.get_observation()

    def step(self, action: int):
        mapped_action = self.action_permutation[action]
        dx, dy = self.ACTIONS[mapped_action]
        ax, ay = self.agent_pos
        nx = ax + dx
        ny = ay + dy

        done = False
        death_by_starvation = False

        self.steps_alive += 1
        reward = 0.0

        # survival cost every step
        if self.use_survival:
            self.energy -= self.step_energy_cost

        if 0 <= nx < self.width and 0 <= ny < self.height:
            target_tile = self.grid[ny][nx]

            ate_food1 = False
            ate_food2 = False

            if target_tile == FOOD1:
                self.food_eaten += 1
                delta = self.food1_energy
                reward += delta
                ate_food1 = True

                if self.use_survival:
                    self.energy += self.food1_energy

            elif target_tile == FOOD2:
                self.food_eaten += 1
                delta = self.food2_energy
                reward += delta
                ate_food2 = True

                if self.use_survival:
                    self.energy += self.food2_energy

            self.grid[ay][ax] = EMPTY
            self.agent_pos = (nx, ny)
            self.grid[ny][nx] = AGENT

            if ate_food1:
                self._respawn_food(FOOD1)
            elif ate_food2:
                self._respawn_food(FOOD2)

        if self.use_survival:
            self.energy = min(self.energy, self.max_energy)

            if self.end_episode_on_zero_energy and self.energy <= 0.0:
                done = True
                death_by_starvation = True

            self.energy_sum += self.energy

        if self.steps_alive >= self.episode_horizon:
            done = True

        self.episode_return += reward

        self.recompute_corner_smells()
        obs = self.get_observation()
        avg_energy = self.energy_sum / max(self.steps_alive + 1, 1)
        info = {
            "food_eaten": self.food_eaten,
            "steps_alive": self.steps_alive,
            "episode_return": self.episode_return,
            "energy": self.energy,
            "avg_energy": avg_energy,
            "death_by_starvation": death_by_starvation,
        }
        return obs, reward, done, info

    def get_observation(self) -> np.ndarray:
        ax, ay = self.agent_pos
        vision = np.zeros((5, 3, 3), dtype=np.float32)

        for dy in range(-1, 2):
            for dx in range(-1, 2):
                gx = ax + dx
                gy = ay + dy

                vx = dx + 1
                vy = dy + 1

                if 0 <= gx < self.width and 0 <= gy < self.height:
                    tile = self.grid[gy][gx]
                else:
                    tile = WALL

                vision[tile, vy, vx] = 1.0

        flat = vision.reshape(-1)

        smell = np.array(
            self.corner_smell_food1 + self.corner_smell_food2,
            dtype=np.float32
        )
        smell = smell / max(self.smell_max_value, 1e-6)

        obs = np.concatenate([flat, smell], axis=0)
        obs = obs[self.observation_permutation]
        return obs

    @property
    def obs_size(self) -> int:
        return 3 * 3 * 5 + 8

    @property
    def action_size(self) -> int:
        return 4

    def draw(self, screen, font):
        screen.fill(COLORS["BG"])

        for y in range(self.height):
            for x in range(self.width):
                tile = self.grid[y][x]
                rect = pygame.Rect(
                    x * CELL_SIZE,
                    y * CELL_SIZE,
                    CELL_SIZE,
                    CELL_SIZE,
                )
                pygame.draw.rect(screen, COLORS[tile], rect)
                pygame.draw.rect(screen, COLORS["GRID"], rect, 1)

        text = font.render(
            f"Food eaten: {self.food_eaten}  "
            f"F1: {self.food1_energy}  F2: {self.food2_energy}",
            True,
            (255, 255, 255),
        )

        screen.blit(text, (10, 10))

        self.draw_sensor_panel(screen, font)

    def get_local_vision_tiles(self):
        """
        Returns a 3x3 array of tile ids centered on the agent.
        Out-of-bounds cells are WALL.
        """
        ax, ay = self.agent_pos
        vision_tiles = np.zeros((3, 3), dtype=np.int32)

        for dy in range(-1, 2):
            for dx in range(-1, 2):
                gx = ax + dx
                gy = ay + dy

                vx = dx + 1
                vy = dy + 1

                if 0 <= gx < self.width and 0 <= gy < self.height:
                    tile = self.grid[gy][gx]
                else:
                    tile = WALL

                vision_tiles[vy, vx] = tile

        return vision_tiles

    def draw_sensor_panel(self, screen, font):
        panel_x = self.width * CELL_SIZE + PANEL_MARGIN
        panel_y = PANEL_MARGIN

        # Title
        title = font.render("Agent Vision (3x3)", True, (255, 255, 255))
        screen.blit(title, (panel_x, panel_y))

        vision_tiles = self.get_local_vision_tiles()

        grid_y_start = panel_y + 35

        for y in range(3):
            for x in range(3):
                tile = int(vision_tiles[y, x])

                rect = pygame.Rect(
                    panel_x + x * SENSOR_CELL_SIZE,
                    grid_y_start + y * SENSOR_CELL_SIZE,
                    SENSOR_CELL_SIZE,
                    SENSOR_CELL_SIZE,
                )

                pygame.draw.rect(screen, COLORS[tile], rect)
                pygame.draw.rect(screen, COLORS["GRID"], rect, 2)

        # Labels
        legend_y = grid_y_start + SENSOR_GRID_SIZE + 15
        labels = [
            ("EMPTY", EMPTY),
            ("AGENT", AGENT),
            ("FOOD1", FOOD1),
            ("FOOD2", FOOD2),
            ("WALL", WALL),
        ]

        for i, (name, tile_id) in enumerate(labels):
            y = legend_y + i * 24

            swatch = pygame.Rect(panel_x, y, 18, 18)
            pygame.draw.rect(screen, COLORS[tile_id], swatch)
            pygame.draw.rect(screen, COLORS["GRID"], swatch, 1)

            label = font.render(name, True, (255, 255, 255))
            screen.blit(label, (panel_x + 28, y - 2))

        smell_panel_y = legend_y + len(labels) * 24 + 30
        self.draw_smell_panel(screen, font, panel_x, smell_panel_y)
# ============================================================
# Actor-Critic Network
# ============================================================

class ActorCritic(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, net_cfg: NetworkConfig):
        super().__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.hidden_sizes = list(net_cfg.hidden_sizes)
        self.activation_name = net_cfg.activation

        if net_cfg.activation == "relu":
            act_cls = nn.ReLU
            self.hidden_gain = math.sqrt(2.0)
        else:
            act_cls = nn.Tanh
            self.hidden_gain = 5.0 / 3.0

        layers = []
        prev = input_dim
        self.hidden_linears = nn.ModuleList()
        self.last_hidden_activations = []

        for h in self.hidden_sizes:
            linear = nn.Linear(prev, h)
            self._init_hidden_linear(linear)
            self.hidden_linears.append(linear)
            layers.append(linear)
            layers.append(act_cls())
            prev = h

        self.shared = nn.Sequential(*layers)

        self.policy_head = nn.Linear(prev, action_dim)
        self.value_head = nn.Linear(prev, 1)

        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.constant_(self.policy_head.bias, 0.0)

        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.constant_(self.value_head.bias, 0.0)

    def _init_hidden_linear(self, layer: nn.Linear):
        nn.init.kaiming_uniform_(
            layer.weight,
            a=0.0,
            mode="fan_in",
            nonlinearity="relu" if self.activation_name == "relu" else "linear",
        )
        nn.init.constant_(layer.bias, 0.0)

    def reinitialize_neuron(self, layer_name: str, neuron_idx: int):
        layer = self.get_named_linear_layer(layer_name)

        with torch.no_grad():
            nn.init.kaiming_uniform_(
                layer.weight[neuron_idx:neuron_idx + 1],
                a=0.0,
                mode="fan_in",
                nonlinearity="relu" if self.activation_name == "relu" else "linear",
            )
            layer.bias[neuron_idx] = 0.0
    @staticmethod
    def effective_rank(matrix: torch.Tensor) -> float:
        """
        Entropy-based effective rank of a representation matrix.
        Higher means more diverse activations.
        """
        if matrix.ndim != 2 or matrix.shape[0] < 2:
            return 0.0

        matrix = matrix - matrix.mean(dim=0, keepdim=True)
        s = torch.linalg.svdvals(matrix)
        s = s[s > 1e-12]

        if s.numel() == 0:
            return 0.0

        p = s / s.sum()
        entropy = -(p * torch.log(p + 1e-12)).sum()
        return float(torch.exp(entropy).item())

    @staticmethod
    def stable_rank_99(matrix: torch.Tensor) -> float:
        """
        Number of singular values needed to explain 99% of singular-value mass.
        This tracks the paper's representation-diversity diagnostic.
        """
        if matrix.ndim != 2 or matrix.shape[0] < 2:
            return 0.0

        matrix = matrix - matrix.mean(dim=0, keepdim=True)
        s = torch.linalg.svdvals(matrix)
        s = s[s > 1e-12]

        if s.numel() == 0:
            return 0.0

        frac = torch.cumsum(s, dim=0) / s.sum()
        k = int(torch.searchsorted(frac, torch.tensor(0.99, device=s.device)).item()) + 1
        return float(k)

    def zero_outgoing_to_neuron_in_next_layer(self, hidden_layer_idx: int, neuron_idx: int):
        with torch.no_grad():
            if hidden_layer_idx + 1 < len(self.hidden_linears):
                next_layer = self.hidden_linears[hidden_layer_idx + 1]
                next_layer.weight[:, neuron_idx] = 0.0
            else:
                self.policy_head.weight[:, neuron_idx] = 0.0
                self.value_head.weight[:, neuron_idx] = 0.0

    def forward(self, x):
        z = x
        self.last_hidden_activations = []

        for module in self.shared:
            z = module(z)
            if isinstance(module, (nn.Tanh, nn.ReLU)):
                self.last_hidden_activations.append(z.detach())

        logits = self.policy_head(z)
        value = self.value_head(z).squeeze(-1)
        return logits, value




    def act(self, obs: torch.Tensor):
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy, value

    # ----------------------------
    # Direct inspection / editing API
    # ----------------------------

    def get_named_linear_layer(self, name: str) -> nn.Linear:
        if name == "policy":
            return self.policy_head
        if name == "value":
            return self.value_head
        if name.startswith("hidden:"):
            idx = int(name.split(":")[1])
            return self.hidden_linears[idx]
        raise ValueError(f"Unknown layer name: {name}")


# ============================================================
# Rollout Buffer
# ============================================================

class RolloutBuffer:
    def __init__(self):
        self.obs = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

        self.advantages = None
        self.returns = None

    def clear(self):
        self.__init__()

    def add(self, obs, action, log_prob, reward, done, value):
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def compute_returns_and_advantages(self, last_value: float, gamma: float, gae_lambda: float):
        n = len(self.rewards)
        advantages = np.zeros(n, dtype=np.float32)
        returns = np.zeros(n, dtype=np.float32)

        gae = 0.0
        for t in reversed(range(n)):
            if t == n - 1:
                next_non_terminal = 1.0 - float(self.dones[t])
                next_value = last_value
            else:
                next_non_terminal = 1.0 - float(self.dones[t])
                next_value = self.values[t + 1]

            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            advantages[t] = gae

        returns = advantages + np.array(self.values, dtype=np.float32)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        self.advantages = advantages
        self.returns = returns

class RunLogger:
    def __init__(
        self,
        env_cfg: EnvConfig,
        net_cfg: NetworkConfig,
        ppo_cfg: PPOConfig,
        run_dir="grid_runs",
        run_name: str = "run",
    ):
        os.makedirs(run_dir, exist_ok=True)

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = run_dir
        self.run_name = run_name

        base = f"{self.run_name}"

        self.csv_path = os.path.join(run_dir, f"{base}_episodes.csv")
        self.config_path = os.path.join(run_dir, f"{base}_config.json")

        config_data = {
            "timestamp": self.timestamp,
            "run_name": self.run_name,
            "env_config": asdict(env_cfg),
            "network_config": asdict(net_cfg),
            "ppo_config": asdict(ppo_cfg),
        }
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2)

        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "episode",
                "env_step",
                "world_id",
                "episode_in_world",
                "reward",
                "final_energy",
                "avg_energy",
                "steps_alive",
                "death_by_starvation",
                "update_index",
            ])

        self.world_changes_path = os.path.join(run_dir, f"{base}_world_changes.csv")

        with open(self.world_changes_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "episode",
                "env_step",
                "from_world_id",
                "to_world_id",
                "update_index",
            ])

        self.activation_path = os.path.join(run_dir, f"{base}_activations.csv")

        with open(self.activation_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "run_name",
                "update_index",
                "env_step",
                "world_id",
                "layer_idx",
                "mean_abs_activation",
                "mean_activation",
                "std_activation",
                "dormant_fraction",
                "effective_rank",
                "stable_rank_99",
                "num_reinitialized_total",
            ])

    def log_activation_metrics(
        self,
        run_name: str,
        update_index: int,
        env_step: int,
        world_id: int,
        layer_idx: int,
        mean_abs_activation: float,
        mean_activation: float,
        std_activation: float,
        dormant_fraction: float,
        effective_rank: float,
        stable_rank_99: float,
        num_reinitialized_total: int,
    ):
        with open(self.activation_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                run_name,
                update_index,
                env_step,
                world_id,
                layer_idx,
                mean_abs_activation,
                mean_activation,
                std_activation,
                dormant_fraction,
                effective_rank,
                stable_rank_99,
                num_reinitialized_total,
            ])

    def log_world_change(
            self,
            episode: int,
            env_step: int,
            from_world_id: int,
            to_world_id: int,
            update_index: int,
    ):
        with open(self.world_changes_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                episode,
                env_step,
                from_world_id,
                to_world_id,
                update_index,
            ])

    def log_episode(
            self,
            episode: int,
            env_step: int,
            world_id: int,
            episode_in_world: int,
            reward: float,
            final_energy: float,
            avg_energy: float,
            steps_alive: int,
            death_by_starvation: bool,
            update_index: int,
    ):
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                episode,
                env_step,
                world_id,
                episode_in_world,
                reward,
                final_energy,
                avg_energy,
                steps_alive,
                int(death_by_starvation),
                update_index,
            ])
# ============================================================
# PPO Trainer
# ============================================================

class PPOTrainer:
    def __init__(
        self,
        env: GridWorld,
        net_cfg: NetworkConfig,
        ppo_cfg: PPOConfig,
        env_cfg: EnvConfig,
        training_worlds: List[WorldSpec],
        logger: Optional[RunLogger] = None
    ):
        self.env = env
        self.env_cfg = env_cfg
        self.net_cfg = net_cfg
        self.cfg = ppo_cfg
        self.logger = logger
        self.current_update_index = 0
        self.device = torch.device(ppo_cfg.device)

        self.total_env_steps = 0

        self.training_worlds = training_worlds

        random.seed(ppo_cfg.seed)
        np.random.seed(ppo_cfg.seed)
        torch.manual_seed(ppo_cfg.seed)

        self.model = ActorCritic(env.obs_size, env.action_size, net_cfg).to(self.device)
        weight_decay = ppo_cfg.l2_coefficient if ppo_cfg.use_l2_regularization else 0.0
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=ppo_cfg.learning_rate,
            weight_decay=weight_decay,
            betas=(ppo_cfg.adam_beta1, ppo_cfg.adam_beta2),
        )

        self.buffer = RolloutBuffer()
        self.episodes_per_world = [0 for _ in self.training_worlds]
        self.episode_count = 0
        self.best_episode_avg_energy = -1

        self.gradient_step_count = 0
        self.num_reinitialized_total = 0

        # One EMA utility vector per hidden layer
        self.hidden_utilities = []
        for linear in self.model.hidden_linears:
            num_units = linear.out_features
            util = torch.zeros(num_units, dtype=torch.float32, device=self.device)
            self.hidden_utilities.append(util)

        self.hidden_ages = []
        self.hidden_replacement_residuals = []

        for linear in self.model.hidden_linears:
            num_units = linear.out_features
            self.hidden_ages.append(torch.zeros(num_units, dtype=torch.long, device=self.device))
            self.hidden_replacement_residuals.append(0.0)

        self.current_world_idx = 0
        self.current_world_steps = 0

        self.current_obs = self.env.reset(
            world_spec=self.training_worlds[self.current_world_idx]
        )

    def _sample_recent_observations_for_activation_metrics(self):
        """
        Uses the most recent rollout buffer as a representative sample.
        """
        if len(self.buffer.obs) == 0:
            return None

        obs_np = np.array(self.buffer.obs, dtype=np.float32)
        n = obs_np.shape[0]
        sample_n = min(self.cfg.activation_sample_size, n)

        if sample_n < n:
            idx = np.random.choice(n, size=sample_n, replace=False)
            obs_np = obs_np[idx]

        return obs_np

    def log_activation_metrics(self):
        if self.logger is None:
            return

        obs_np = self._sample_recent_observations_for_activation_metrics()
        if obs_np is None:
            return

        x = torch.tensor(obs_np, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            _ = self.model.forward(x)
            acts = self.model.last_hidden_activations

        current_world = self._current_world()

        for layer_idx, a in enumerate(acts):
            # a shape: [batch, units]
            mean_abs_per_unit = a.abs().mean(dim=0)
            dormant_fraction = float(
                (mean_abs_per_unit < self.cfg.dormant_threshold).float().mean().item()
            )

            mean_abs_activation = float(a.abs().mean().item())
            mean_activation = float(a.mean().item())
            std_activation = float(a.std().item())

            effective_rank = self.model.effective_rank(a)
            stable_rank_99 = self.model.stable_rank_99(a)

            self.logger.log_activation_metrics(
                run_name=self.logger.run_name,
                update_index=self.current_update_index,
                env_step=self.total_env_steps,
                world_id=int(current_world.world_id),
                layer_idx=layer_idx,
                mean_abs_activation=mean_abs_activation,
                mean_activation=mean_activation,
                std_activation=std_activation,
                dormant_fraction=dormant_fraction,
                effective_rank=effective_rank,
                stable_rank_99=stable_rank_99,
                num_reinitialized_total=int(self.num_reinitialized_total),
            )

    def measure_dormant_fraction(self, sample_obs: np.ndarray, threshold: float = 0.01):
        x = torch.tensor(sample_obs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            _ = self.model.forward(x)
            acts = self.model.last_hidden_activations

        total = 0
        dormant = 0
        for a in acts:
            mean_abs = a.abs().mean(dim=0)
            dormant += int((mean_abs < threshold).sum().item())
            total += mean_abs.numel()

        return dormant / max(total, 1)

    def _current_world(self) -> WorldSpec:
        return self.training_worlds[self.current_world_idx]

    def _reset_episode_in_current_world(self):
        self.current_obs = self.env.reset(
            world_spec=self._current_world()
        )

    def _switch_world(self):
        old_world_idx = self.current_world_idx
        self.current_world_idx = (self.current_world_idx + 1) % len(self.training_worlds)
        self.current_world_steps = 0

        if self.logger is not None:
            self.logger.log_world_change(
                episode=self.episode_count,
                env_step=self.total_env_steps,
                from_world_id=int(self.training_worlds[old_world_idx].world_id),
                to_world_id=int(self.training_worlds[self.current_world_idx].world_id),
                update_index=self.current_update_index,
            )

        self.current_obs = self.env.reset(
            world_spec=self._current_world()
        )

    def _finish_episode_and_log(self, info):
        self.episode_count += 1
        self.episodes_per_world[self.current_world_idx] += 1
        self.best_episode_avg_energy = max(self.best_episode_avg_energy, float(info["avg_energy"]))

        if self.logger is not None:
            self.logger.log_episode(
                episode=self.episode_count,
                env_step=self.total_env_steps,
                world_id=int(self._current_world().world_id),
                episode_in_world=self.episodes_per_world[self.current_world_idx],
                reward=float(info["episode_return"]),
                final_energy=float(info["energy"]),
                avg_energy=float(info["avg_energy"]),
                steps_alive=int(info["steps_alive"]),
                death_by_starvation=bool(info["death_by_starvation"]),
                update_index=self.current_update_index,
            )

    def average_weight_magnitude(self):
        total = 0.0
        count = 0
        with torch.no_grad():
            for p in self.model.parameters():
                if p.ndim >= 1:
                    total += p.abs().sum().item()
                    count += p.numel()
        return total / max(count, 1)


    def update_hidden_utilities(self):
        if not self.cfg.use_continual_backprop:
            return

        if len(self.model.last_hidden_activations) != len(self.hidden_utilities):
            return

        decay = self.cfg.cbp_decay

        for layer_idx, acts in enumerate(self.model.last_hidden_activations):
            # acts shape: [batch, units]
            mean_abs_act = acts.abs().mean(dim=0).detach()

            # outgoing weights from this hidden layer
            if layer_idx + 1 < len(self.model.hidden_linears):
                next_weight = self.model.hidden_linears[layer_idx + 1].weight.detach()
                outgoing_mag = next_weight.abs().sum(dim=0)
            else:
                # last hidden layer feeds both policy and value heads
                policy_mag = self.model.policy_head.weight.detach().abs().sum(dim=0)
                value_mag = self.model.value_head.weight.detach().abs().sum(dim=0)
                outgoing_mag = policy_mag + value_mag

            instant_utility = mean_abs_act * outgoing_mag

            self.hidden_utilities[layer_idx] = (
                    decay * self.hidden_utilities[layer_idx]
                    + (1.0 - decay) * instant_utility
            )

            # all units age by one update
            self.hidden_ages[layer_idx] += 1

    def maybe_apply_continual_backprop(self):
        if not self.cfg.use_continual_backprop:
            return

        for layer_idx, util in enumerate(self.hidden_utilities):
            ages = self.hidden_ages[layer_idx]
            mature_mask = ages >= self.cfg.cbp_min_steps_before_reinit

            mature_indices = torch.nonzero(mature_mask, as_tuple=False).flatten()
            num_mature = int(mature_indices.numel())

            if num_mature == 0:
                continue

            expected = num_mature * self.cfg.cbp_reinit_fraction + self.hidden_replacement_residuals[layer_idx]
            num_reinit = int(expected)
            self.hidden_replacement_residuals[layer_idx] = expected - num_reinit

            if num_reinit <= 0:
                continue

            mature_utils = util[mature_indices]
            _, local_idx = torch.topk(mature_utils, k=min(num_reinit, num_mature), largest=False)
            chosen = mature_indices[local_idx]

            for idx_tensor in chosen:
                neuron_idx = int(idx_tensor.item())

                self.model.reinitialize_neuron(f"hidden:{layer_idx}", neuron_idx)
                self.model.zero_outgoing_to_neuron_in_next_layer(layer_idx, neuron_idx)

                self.hidden_utilities[layer_idx][neuron_idx] = 0.0
                self.hidden_ages[layer_idx][neuron_idx] = 0

                self._clear_optimizer_state_for_neuron(layer_idx, neuron_idx)

                self.num_reinitialized_total += 1

    def _clear_optimizer_state_for_neuron(self, hidden_layer_idx: int, neuron_idx: int):
        layer = self.model.hidden_linears[hidden_layer_idx]

        # Clear optimizer state for incoming weights and bias of this neuron
        for param, row_idx in [(layer.weight, neuron_idx), (layer.bias, neuron_idx)]:
            state = self.optimizer.state.get(param, None)
            if state is None:
                continue

            if "exp_avg" in state:
                if param.ndim == 2:
                    state["exp_avg"][row_idx].zero_()
                else:
                    state["exp_avg"][row_idx] = 0.0

            if "exp_avg_sq" in state:
                if param.ndim == 2:
                    state["exp_avg_sq"][row_idx].zero_()
                else:
                    state["exp_avg_sq"][row_idx] = 0.0

    def collect_rollout(self, render=False, screen=None, clock=None, font=None, render_every=10):
        self.buffer.clear()

        obs = self.current_obs

        for step_idx in range(self.cfg.steps_per_update):
            if render and step_idx % render_every == 0:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()

                self.env.draw(screen, font)
                pygame.display.flip()
                clock.tick(FPS)

            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

            with torch.no_grad():
                action_t, log_prob_t, value_t = self.model.act(obs_t)

            action = int(action_t.item())
            log_prob = float(log_prob_t.item())
            value = float(value_t.item())

            next_obs, reward, done, info = self.env.step(action)

            self.total_env_steps += 1
            self.current_world_steps += 1

            self.buffer.add(obs, action, log_prob, reward, done, value)
            obs = next_obs

            # End of episode: reset in the SAME current world
            if done:
                self._finish_episode_and_log(info)
                self._reset_episode_in_current_world()
                obs = self.current_obs

            # End of world phase: switch to NEXT world in the cycle
            if self.current_world_steps >= self.cfg.steps_per_world:
                self._switch_world()
                obs = self.current_obs

            if self.total_env_steps >= self.cfg.total_env_steps_target:
                break

        self.current_obs = obs

        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            _, last_value_t = self.model.forward(obs_t)
        last_value = float(last_value_t.item())

        if len(self.buffer.rewards) > 0:
            self.buffer.compute_returns_and_advantages(
                last_value=last_value,
                gamma=self.cfg.gamma,
                gae_lambda=self.cfg.gae_lambda,
            )

    def update(self):
        obs = torch.tensor(np.array(self.buffer.obs), dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.array(self.buffer.actions), dtype=torch.long, device=self.device)
        old_log_probs = torch.tensor(np.array(self.buffer.log_probs), dtype=torch.float32, device=self.device)
        advantages = torch.tensor(self.buffer.advantages, dtype=torch.float32, device=self.device)
        returns = torch.tensor(self.buffer.returns, dtype=torch.float32, device=self.device)

        n = obs.shape[0]
        indices = np.arange(n)

        stats = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "num_updates": 0,
        }

        for _ in range(self.cfg.ppo_epochs):
            np.random.shuffle(indices)

            for start in range(0, n, self.cfg.minibatch_size):
                end = start + self.cfg.minibatch_size
                mb_idx = indices[start:end]

                mb_obs = obs[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_advantages = advantages[mb_idx]
                mb_returns = returns[mb_idx]

                new_log_probs, entropy, values = self.model.evaluate_actions(mb_obs, mb_actions)
                self.update_hidden_utilities()

                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.cfg.clip_epsilon,
                    1.0 + self.cfg.clip_epsilon
                ) * mb_advantages

                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = torch.mean((mb_returns - values) ** 2)
                entropy_bonus = entropy.mean()

                loss = (
                        policy_loss
                        + self.cfg.value_coef * value_loss
                        - self.cfg.entropy_coef * entropy_bonus
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()
                self.gradient_step_count += 1
                self.maybe_apply_continual_backprop()

                stats["policy_loss"] += float(policy_loss.item())
                stats["value_loss"] += float(value_loss.item())
                stats["entropy"] += float(entropy_bonus.item())
                stats["num_updates"] += 1

        for k in ("policy_loss", "value_loss", "entropy"):
            stats[k] /= max(stats["num_updates"], 1)

        return stats

    def train(self, render=False):
        screen = clock = font = None
        if render:
            screen, clock, font = init_training_viewer()

        update = 0
        while self.total_env_steps < self.cfg.total_env_steps_target:
            update += 1
            self.current_update_index = update

            self.collect_rollout(
                render=render,
                screen=screen,
                clock=clock,
                font=font,
                render_every=1,
            )
            stats = self.update()
            if update % self.cfg.activation_log_every_updates == 0 or update == 1:
                self.log_activation_metrics()

            if update % 10 == 0 or update == 1:
                avg_w = self.average_weight_magnitude()
                current_world = self._current_world()

                print(
                    f"Update {update:5d} | "
                    f"EnvSteps {self.total_env_steps:8d}/{self.cfg.total_env_steps_target} | "
                    f"World {current_world.world_id:2d} | "
                    f"WorldSteps {self.current_world_steps:6d}/{self.cfg.steps_per_world} | "
                    f"F1 {current_world.food1_energy:6.2f} | "
                    f"F2 {current_world.food2_energy:6.2f} | "
                    f"Episodes {self.episode_count:5d} | "
                    f"BestAvgEnergy {self.best_episode_avg_energy:6.2f} | "
                    f"PolicyLoss {stats['policy_loss']:.4f} | "
                    f"ValueLoss {stats['value_loss']:.4f} | "
                    f"Entropy {stats['entropy']:.4f} | "
                    f"Avg|W| {avg_w:.6f}"
                )

        print(f"\nTraining complete: reached {self.episode_count} episodes.\n")

    def save(self, path="ppo_gridworld.pt"):
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "net_hidden_sizes": self.net_cfg.hidden_sizes,
            },
            path
        )


# ============================================================
# Example weight/neuron editing helpers
# ============================================================

def run_experiment(
    run_name: str,
    use_l2: bool,
    use_cb: bool,
    seed: int,
    training_worlds: List[WorldSpec],
):
    adam_beta1 = 0.9
    adam_beta2 = 0.999

    env_cfg = EnvConfig(
        width=30,
        height=30,
        num_food1=30,
        num_food2=30,
        food1_energy=int(GLOBAL_ENERGY_E),
        food2_energy=int(GLOBAL_ENERGY_E),
        episode_horizon=200,
    )

    net_cfg = NetworkConfig(
        hidden_sizes=[256, 256],
        activation="relu",
    )

    steps_per_world = 100_000

    ppo_cfg = PPOConfig(
        total_env_steps_target=NUM_TRAIN_WORLDS * steps_per_world,
        steps_per_update=2048,
        steps_per_world=steps_per_world,
        ppo_epochs=10,
        minibatch_size=256,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        learning_rate=1e-4,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        use_l2_regularization=use_l2,
        l2_coefficient=1e-4,
        use_continual_backprop=use_cb,
        cbp_decay=0.99,
        cbp_reinit_fraction=1e-4,
        cbp_min_steps_before_reinit=10_000,
        device="cpu",
        seed=seed,
        adam_beta1=adam_beta1,
        adam_beta2=adam_beta2,
    )
    env = GridWorld(env_cfg, seed=seed)
    run_logger = RunLogger(env_cfg, net_cfg, ppo_cfg, run_dir="grid_runs", run_name=run_name)

    trainer = PPOTrainer(
        env=env,
        net_cfg=net_cfg,
        ppo_cfg=ppo_cfg,
        env_cfg=env_cfg,
        training_worlds=training_worlds,
        logger=run_logger,
    )

    print(f"[{run_name}] Episode log file: {run_logger.csv_path}")
    print(f"[{run_name}] Config file: {run_logger.config_path}")
    print(f"[{run_name}] L2 enabled: {ppo_cfg.use_l2_regularization}")
    print(f"[{run_name}] CB enabled: {ppo_cfg.use_continual_backprop}")
    print(f"[{run_name}] Number of training worlds: {len(training_worlds)}")
    print(f"[{run_name}] Steps per world: {ppo_cfg.steps_per_world}")

    trainer.train(render=False)
    trainer.save(os.path.join("grid_runs", f"{run_name}.pt"))
# ============================================================
# Main
# ============================================================
from multiprocessing import Process

def main():
    env_cfg = EnvConfig(
        width=30,
        height=30,
        num_food1=30,
        num_food2=30,
        food1_energy=int(GLOBAL_ENERGY_E),
        food2_energy=int(GLOBAL_ENERGY_E),
        episode_horizon=200,
    )

    jobs = [
        ("ppo", False, False),
        ("ppo_l2", True, False),
        ("ppo_cb", False, True),
        ("ppo_l2_cb", True, True),
    ]

    num_trials = 10
    base_seed = 1

    start_trial = 1  # human-readable trial number
    end_trial_inclusive  = num_trials  # human-readable trial number, inclusive

    for trial_idx in range(start_trial - 1, end_trial_inclusive):
        trial_num = trial_idx + 1
        print(f"\n==============================")
        print(f"Starting trial {trial_num}/{num_trials}")
        print(f"==============================\n")

        # Make a separate world set for this trial.
        # All 4 methods in the same trial get the SAME worlds.
        training_worlds = generate_training_worlds(
            cfg=env_cfg,
            num_worlds=NUM_TRAIN_WORLDS,
            E=GLOBAL_ENERGY_E,
            seed=TRAIN_WORLD_SEED + trial_idx,
        )

        processes = []

        for method_name, use_l2, use_cb in jobs:
            seed = base_seed + trial_idx
            run_name = f"{method_name}_trial_{trial_num:02d}"

            p = Process(
                target=run_experiment,
                args=(run_name, use_l2, use_cb, seed, training_worlds),
            )
            p.start()
            processes.append((run_name, p))

        # Wait for all 4 methods in this trial to finish
        failed = []
        for run_name, p in processes:
            p.join()
            if p.exitcode != 0:
                failed.append((run_name, p.exitcode))

        if failed:
            print(f"\nTrial {trial_num} finished with failures:")
            for run_name, exitcode in failed:
                print(f"  {run_name}: exit code {exitcode}")
        else:
            print(f"\nTrial {trial_num} complete: all 4 methods finished successfully.")

    print("\nAll trials finished.")



def init_training_viewer():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("PPO Training Viewer")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 28)
    return screen, clock, font


if __name__ == "__main__":
    main()