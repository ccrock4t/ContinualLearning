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

GRID_WIDTH = 20
GRID_HEIGHT = 20
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

FOOD_ENERGY = 20
AGENT_MAX_ENERGY = FOOD_ENERGY

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
class EnvConfig:
    width: int = 20
    height: int = 20
    num_food1: int = 20
    num_food2: int = 12
    max_energy: int = AGENT_MAX_ENERGY
    food_energy: int = FOOD_ENERGY


@dataclass
class NetworkConfig:
    hidden_sizes: List[int] = field(default_factory=lambda: [128, 128])
    activation: str = "tanh"   # "tanh" or "relu"


@dataclass
class PPOConfig:
    rollout_steps: int = 2048
    ppo_epochs: int = 10
    minibatch_size: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    learning_rate: float = 3e-4
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5

    # L2 regularization
    use_l2_regularization: bool = False
    l2_coefficient: float = 1e-5

    # Continual backpropagation
    use_continual_backprop: bool = False
    cbp_decay: float = 0.99
    cbp_reinit_fraction: float = 0.01
    cbp_min_steps_before_reinit: int = 100

    # training
    total_updates: int = 300
    device: str = "cpu"
    seed: int = 1

# ============================================================
# Environment
# ============================================================

class GridWorld:
    """
    PPO-controlled grid world.
    Reward is 0 on all intermediate steps.
    At episode end (energy <= 0), reward = total number of food eaten in the episode.
    """

    ACTIONS = [
        (0, -1),   # up
        (0, 1),    # down
        (-1, 0),   # left
        (1, 0),    # right
    ]

    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        self.width = cfg.width
        self.height = cfg.height
        self.num_food1 = cfg.num_food1
        self.num_food2 = cfg.num_food2
        self.max_energy = cfg.max_energy
        self.food_energy = cfg.food_energy

        self.grid = [[EMPTY for _ in range(self.width)] for _ in range(self.height)]
        self.agent_pos = (0, 0)
        self.agent_energy = self.max_energy
        self.food_eaten = 0
        self.corner_smell_food1 = [0.0, 0.0, 0.0, 0.0]
        self.corner_smell_food2 = [0.0, 0.0, 0.0, 0.0]
        self.smell_max_value = 1.0

        self.reset()

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


    def reset(self):
        self.grid = [[EMPTY for _ in range(self.width)] for _ in range(self.height)]
        self.agent_energy = self.max_energy
        self.food_eaten = 0
        self.steps_alive = 0

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

    def _random_empty_cell(self):
        while True:
            x = random.randrange(self.width)
            y = random.randrange(self.height)
            if self.grid[y][x] == EMPTY:
                return x, y

    def step(self, action: int):
        dx, dy = self.ACTIONS[action]
        ax, ay = self.agent_pos
        nx = ax + dx
        ny = ay + dy

        reward = 0.0
        done = False

        # Lose 1 energy every step
        self.agent_energy -= 1

        # add steps alive
        self.steps_alive += 1

        # Move only if in bounds
        if 0 <= nx < self.width and 0 <= ny < self.height:
            target_tile = self.grid[ny][nx]

            if target_tile in (FOOD1, FOOD2):
                self.food_eaten += 1
                self.agent_energy = self.max_energy

            self.grid[ay][ax] = EMPTY
            self.agent_pos = (nx, ny)
            self.grid[ny][nx] = AGENT

        # Episode ends when energy runs out
        if self.agent_energy <= 0:
            done = True
            reward = float(self.food_eaten)

        self.recompute_corner_smells()
        obs = self.get_observation()
        info = {
            "food_eaten": self.food_eaten,
            "energy": self.agent_energy,
            "steps_alive": self.steps_alive,
        }
        return obs, reward, done, info

    def get_observation(self) -> np.ndarray:
        """
        Local 3x3 vision centered on the agent, one-hot encoded over 5 tile types
        (EMPTY, AGENT, FOOD1, FOOD2, WALL), plus normalized energy.
        Shape = (3 * 3 * 5 + 1,) = 46
        """
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
        energy = np.array([self.agent_energy / float(self.max_energy)], dtype=np.float32)
        return np.concatenate([flat, energy], axis=0)

    @property
    def obs_size(self) -> int:
        return 3 * 3 * 5 + 1

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
            f"Energy: {self.agent_energy}/{self.max_energy}  Food eaten: {self.food_eaten}",
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
    """
    Shared torso + policy head + value head.
    Provides direct functions to inspect and modify layers, neurons, and connections.
    """

    def __init__(self, input_dim: int, action_dim: int, net_cfg: NetworkConfig):
        super().__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.hidden_sizes = list(net_cfg.hidden_sizes)

        if net_cfg.activation == "relu":
            act_cls = nn.ReLU
        else:
            act_cls = nn.Tanh

        layers = []
        prev = input_dim
        self.hidden_linears = nn.ModuleList()
        self.last_hidden_activations = []

        for h in self.hidden_sizes:
            linear = nn.Linear(prev, h)
            nn.init.orthogonal_(linear.weight, gain=math.sqrt(2))
            nn.init.constant_(linear.bias, 0.0)
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

    def reinitialize_neuron(self, layer_name: str, neuron_idx: int):
        layer = self.get_named_linear_layer(layer_name)

        with torch.no_grad():
            nn.init.orthogonal_(layer.weight[neuron_idx:neuron_idx + 1], gain=math.sqrt(2))
            layer.bias[neuron_idx] = 0.0

    def zero_outgoing_to_neuron_in_next_layer(self, hidden_layer_idx: int, neuron_idx: int):
        """
        If hidden layer i has neuron k reinitialized, zero its outgoing weights
        in the next linear layer to reduce disruption.
        """
        next_layer = None

        if hidden_layer_idx + 1 < len(self.hidden_linears):
            next_layer = self.hidden_linears[hidden_layer_idx + 1]
        else:
            next_layer = self.policy_head

        with torch.no_grad():
            next_layer.weight[:, neuron_idx] = 0.0

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

    def get_hidden_layer(self, layer_idx: int) -> nn.Linear:
        return self.hidden_linears[layer_idx]

    def get_named_linear_layer(self, name: str) -> nn.Linear:
        if name == "policy":
            return self.policy_head
        if name == "value":
            return self.value_head
        if name.startswith("hidden:"):
            idx = int(name.split(":")[1])
            return self.hidden_linears[idx]
        raise ValueError(f"Unknown layer name: {name}")

    def get_weight(self, layer_name: str, out_idx: int, in_idx: int) -> float:
        layer = self.get_named_linear_layer(layer_name)
        with torch.no_grad():
            return float(layer.weight[out_idx, in_idx].item())

    def set_weight(self, layer_name: str, out_idx: int, in_idx: int, value: float):
        layer = self.get_named_linear_layer(layer_name)
        with torch.no_grad():
            layer.weight[out_idx, in_idx] = value

    def add_to_weight(self, layer_name: str, out_idx: int, in_idx: int, delta: float):
        layer = self.get_named_linear_layer(layer_name)
        with torch.no_grad():
            layer.weight[out_idx, in_idx] += delta

    def zero_weight(self, layer_name: str, out_idx: int, in_idx: int):
        self.set_weight(layer_name, out_idx, in_idx, 0.0)

    def get_bias(self, layer_name: str, neuron_idx: int) -> float:
        layer = self.get_named_linear_layer(layer_name)
        with torch.no_grad():
            return float(layer.bias[neuron_idx].item())

    def set_bias(self, layer_name: str, neuron_idx: int, value: float):
        layer = self.get_named_linear_layer(layer_name)
        with torch.no_grad():
            layer.bias[neuron_idx] = value

    def get_neuron_weights(self, layer_name: str, neuron_idx: int) -> np.ndarray:
        layer = self.get_named_linear_layer(layer_name)
        with torch.no_grad():
            return layer.weight[neuron_idx].detach().cpu().numpy().copy()

    def set_neuron_weights(self, layer_name: str, neuron_idx: int, weights: np.ndarray, bias: Optional[float] = None):
        layer = self.get_named_linear_layer(layer_name)
        weights = np.asarray(weights, dtype=np.float32)
        if weights.shape[0] != layer.weight.shape[1]:
            raise ValueError(f"Expected {layer.weight.shape[1]} incoming weights, got {weights.shape[0]}")
        with torch.no_grad():
            layer.weight[neuron_idx] = torch.tensor(weights, dtype=layer.weight.dtype, device=layer.weight.device)
            if bias is not None:
                layer.bias[neuron_idx] = bias

    def zero_neuron(self, layer_name: str, neuron_idx: int):
        layer = self.get_named_linear_layer(layer_name)
        with torch.no_grad():
            layer.weight[neuron_idx].zero_()
            layer.bias[neuron_idx] = 0.0

    def scale_neuron(self, layer_name: str, neuron_idx: int, factor: float):
        layer = self.get_named_linear_layer(layer_name)
        with torch.no_grad():
            layer.weight[neuron_idx] *= factor
            layer.bias[neuron_idx] *= factor

    def freeze_layer(self, layer_name: str, freeze: bool = True):
        layer = self.get_named_linear_layer(layer_name)
        for p in layer.parameters():
            p.requires_grad = not freeze


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
    def __init__(self, env_cfg: EnvConfig, net_cfg: NetworkConfig, ppo_cfg: PPOConfig, run_dir="runs"):
        os.makedirs(run_dir, exist_ok=True)

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = run_dir

        self.csv_path = os.path.join(run_dir, f"{self.timestamp}_episodes.csv")
        self.config_path = os.path.join(run_dir, f"{self.timestamp}_config.json")

        # Save config immediately
        config_data = {
            "timestamp": self.timestamp,
            "env_config": asdict(env_cfg),
            "network_config": asdict(net_cfg),
            "ppo_config": asdict(ppo_cfg),
        }
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2)

        # Create CSV header
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "episode",
                "reward",
                "food_eaten",
                "steps_alive",
                "update_index",
            ])

    def log_episode(self, episode: int, reward: float, food_eaten: int, steps_alive: int, update_index: int):
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                episode,
                reward,
                food_eaten,
                steps_alive,
                update_index,
            ])
# ============================================================
# PPO Trainer
# ============================================================

class PPOTrainer:
    def __init__(self, env: GridWorld, net_cfg: NetworkConfig, ppo_cfg: PPOConfig, env_cfg: EnvConfig,
                 logger: Optional[RunLogger] = None):
        self.env = env
        self.env_cfg = env_cfg
        self.net_cfg = net_cfg
        self.cfg = ppo_cfg
        self.logger = logger
        self.current_update_index = 0
        self.device = torch.device(ppo_cfg.device)

        random.seed(ppo_cfg.seed)
        np.random.seed(ppo_cfg.seed)
        torch.manual_seed(ppo_cfg.seed)

        self.model = ActorCritic(env.obs_size, env.action_size, net_cfg).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=ppo_cfg.learning_rate)

        self.buffer = RolloutBuffer()

        self.episode_count = 0
        self.best_episode_food = -1

        self.gradient_step_count = 0

        # One EMA utility vector per hidden layer
        self.hidden_utilities = []
        for linear in self.model.hidden_linears:
            num_units = linear.out_features
            util = torch.zeros(num_units, dtype=torch.float32, device=self.device)
            self.hidden_utilities.append(util)

    def update_hidden_utilities(self):
        if not self.cfg.use_continual_backprop:
            return

        if not hasattr(self.model, "last_hidden_activations"):
            return

        if len(self.model.last_hidden_activations) != len(self.hidden_utilities):
            return

        decay = self.cfg.cbp_decay

        for layer_idx, acts in enumerate(self.model.last_hidden_activations):
            # acts shape: [batch, units]
            mean_abs = acts.abs().mean(dim=0).detach()
            self.hidden_utilities[layer_idx] = (
                    decay * self.hidden_utilities[layer_idx]
                    + (1.0 - decay) * mean_abs
            )

    def maybe_apply_continual_backprop(self):
        if not self.cfg.use_continual_backprop:
            return

        if self.gradient_step_count < self.cfg.cbp_min_steps_before_reinit:
            return

        for layer_idx, util in enumerate(self.hidden_utilities):
            num_units = util.shape[0]
            num_reinit = max(1, int(num_units * self.cfg.cbp_reinit_fraction))

            if num_reinit <= 0:
                continue

            # Least-used units
            _, least_used_idx = torch.topk(util, k=num_reinit, largest=False)

            for idx_tensor in least_used_idx:
                neuron_idx = int(idx_tensor.item())

                # Reinitialize incoming weights + bias
                self.model.reinitialize_neuron(f"hidden:{layer_idx}", neuron_idx)

                # Zero outgoing weights into the next layer / policy head
                self.model.zero_outgoing_to_neuron_in_next_layer(layer_idx, neuron_idx)

                # Reset utility so it can compete again fairly
                self.hidden_utilities[layer_idx][neuron_idx] = 0.0

                # Clear Adam state for this neuron's incoming weights/bias
                self._clear_optimizer_state_for_neuron(layer_idx, neuron_idx)

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
        obs = self.env.reset()

        for step_idx in range(self.cfg.rollout_steps):
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

            self.buffer.add(obs, action, log_prob, reward, done, value)
            obs = next_obs

            if done:
                self.episode_count += 1
                self.best_episode_food = max(self.best_episode_food, info["food_eaten"])

                if self.logger is not None:
                    self.logger.log_episode(
                        episode=self.episode_count,
                        reward=float(reward),
                        food_eaten=int(info["food_eaten"]),
                        steps_alive=int(info["steps_alive"]),
                        update_index=self.current_update_index,
                    )

                obs = self.env.reset()

        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            _, last_value_t = self.model.forward(obs_t)
        last_value = float(last_value_t.item())

        self.buffer.compute_returns_and_advantages(
            last_value=last_value,
            gamma=self.cfg.gamma,
            gae_lambda=self.cfg.gae_lambda,
        )
    def compute_l2_penalty(self):
        if not self.cfg.use_l2_regularization:
            return torch.tensor(0.0, device=self.device)

        l2 = torch.tensor(0.0, device=self.device)
        for p in self.model.parameters():
            if p.requires_grad:
                l2 = l2 + torch.sum(p * p)
        return self.cfg.l2_coefficient * l2

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
            "l2": 0.0,
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

                l2_penalty = self.compute_l2_penalty()

                loss = (
                    policy_loss
                    + self.cfg.value_coef * value_loss
                    - self.cfg.entropy_coef * entropy_bonus
                    + l2_penalty
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
                stats["l2"] += float(l2_penalty.item())
                stats["num_updates"] += 1

        for k in ("policy_loss", "value_loss", "entropy", "l2"):
            stats[k] /= max(stats["num_updates"], 1)

        return stats

    def train(self, render=False):
        screen = clock = font = None
        if render:
            screen, clock, font = init_training_viewer()

        for update in range(1, self.cfg.total_updates + 1):
            self.current_update_index = update
            self.collect_rollout(
                render=render,
                screen=screen,
                clock=clock,
                font=font,
                render_every=1,
            )
            stats = self.update()

            if update % 10 == 0 or update == 1:
                print(
                    f"Update {update:4d}/{self.cfg.total_updates} | "
                    f"Episodes {self.episode_count:5d} | "
                    f"BestFood {self.best_episode_food:3d} | "
                    f"PolicyLoss {stats['policy_loss']:.4f} | "
                    f"ValueLoss {stats['value_loss']:.4f} | "
                    f"Entropy {stats['entropy']:.4f} | "
                    f"L2 {stats['l2']:.6f}"
                )

    def save(self, path="ppo_gridworld.pt"):
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "net_hidden_sizes": self.net_cfg.hidden_sizes,
            },
            path
        )

    def load(self, path="ppo_gridworld.pt"):
        data = torch.load(path, map_location=self.device)
        self.model.load_state_dict(data["model_state_dict"])

    def select_action_greedy(self, obs: np.ndarray) -> int:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits, _ = self.model.forward(obs_t)
            action = torch.argmax(logits, dim=-1)
        return int(action.item())

    def select_action_sample(self, obs: np.ndarray) -> int:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action_t, _, _ = self.model.act(obs_t)
        return int(action_t.item())


# ============================================================
# Pygame Viewer
# ============================================================

def watch_trained_agent(trainer: PPOTrainer, env_cfg: EnvConfig, greedy: bool = True):
    pygame.init()
    pygame.display.set_caption("Trained PPO GridWorld")
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 28)

    env = GridWorld(env_cfg)

    obs = env.reset()
    running = True
    episode_idx = 0

    while running:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    obs = env.reset()
                    episode_idx += 1

        if greedy:
            action = trainer.select_action_greedy(obs)
        else:
            action = trainer.select_action_sample(obs)

        obs, reward, done, info = env.step(action)

        env.draw(screen, font)
        pygame.display.flip()

        if done:
            print(f"Viewer episode {episode_idx}: terminal reward = {reward}, food eaten = {info['food_eaten']}")
            obs = env.reset()
            episode_idx += 1

    pygame.quit()


# ============================================================
# Example weight/neuron editing helpers
# ============================================================

def demo_direct_weight_edits(trainer: PPOTrainer):
    model = trainer.model

    # Read one connection
    w = model.get_weight("hidden:0", 0, 0)
    print("Before: hidden:0 weight[0,0] =", w)

    # Set one connection
    model.set_weight("hidden:0", 0, 0, 0.25)

    # Add to one connection
    model.add_to_weight("hidden:0", 0, 1, -0.10)

    # Zero one connection
    model.zero_weight("hidden:0", 0, 2)

    # Read / set bias
    b = model.get_bias("hidden:0", 0)
    print("Before: hidden:0 bias[0] =", b)
    model.set_bias("hidden:0", 0, 0.05)

    # Replace all incoming weights for neuron 1 in hidden layer 0
    incoming_size = model.get_named_linear_layer("hidden:0").weight.shape[1]
    new_weights = np.zeros(incoming_size, dtype=np.float32)
    model.set_neuron_weights("hidden:0", 1, new_weights, bias=0.0)

    # Scale an entire neuron
    model.scale_neuron("hidden:0", 2, 0.5)

    # Zero an entire policy output neuron
    model.zero_neuron("policy", 0)

    print("After: hidden:0 weight[0,0] =", model.get_weight("hidden:0", 0, 0))


# ============================================================
# Main
# ============================================================

def main():
    # ----------------------------
    # Configure environment
    # ----------------------------
    env_cfg = EnvConfig(
        width=20,
        height=20,
        num_food1=20,
        num_food2=12,
        max_energy=20,
        food_energy=20,
    )

    # ----------------------------
    # Configure network
    # Change this to control layers/neurons directly
    # Examples:
    #   [64]
    #   [128, 128]
    #   [256, 128, 64]
    # ----------------------------
    net_cfg = NetworkConfig(
        hidden_sizes=[128, 128],
        activation="tanh",
    )

    # ----------------------------
    # Configure PPO
    # L2 can be enabled/disabled here
    # ----------------------------
    ppo_cfg = PPOConfig(
        rollout_steps=2048,
        ppo_epochs=10,
        minibatch_size=256,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        learning_rate=3e-4,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        use_l2_regularization=False,
        l2_coefficient=1e-6,
        use_continual_backprop=True, 
        cbp_decay=0.99,
        cbp_reinit_fraction=0.01,
        cbp_min_steps_before_reinit=100,
        total_updates=300,
        device="cpu",
        seed=1,
    )

    env = GridWorld(env_cfg)
    run_logger = RunLogger(env_cfg, net_cfg, ppo_cfg, run_dir="runs")
    print("Episode log file:", run_logger.csv_path)
    print("Config file:", run_logger.config_path)
    trainer = PPOTrainer(env, net_cfg, ppo_cfg, env_cfg=env_cfg, logger=run_logger)

    print("Observation size:", env.obs_size)
    print("Action size:", env.action_size)
    print("Hidden sizes:", net_cfg.hidden_sizes)
    print("L2 enabled:", ppo_cfg.use_l2_regularization)
    print("L2 coefficient:", ppo_cfg.l2_coefficient)
    print("Continual backprop enabled:", ppo_cfg.use_continual_backprop)
    print("CBP reinit fraction:", ppo_cfg.cbp_reinit_fraction)

    # Optional: direct surgery on the network before training
    # demo_direct_weight_edits(trainer)

    # Train PPO
    trainer.train(render=True)

    # Save model
    trainer.save("ppo_gridworld.pt")

    # Watch trained agent
    watch_trained_agent(trainer, env_cfg, greedy=True)

def init_training_viewer():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("PPO Training Viewer")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 28)
    return screen, clock, font

if __name__ == "__main__":
    main()