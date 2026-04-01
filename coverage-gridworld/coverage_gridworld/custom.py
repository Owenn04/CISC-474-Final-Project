"""Custom observation and reward logic for Coverage Gridworld.

This module is the project-side extension point used by the immutable environment.
The environment calls ``observation()`` and ``reward()`` directly, so all richer
observation modes and reward shaping must be implemented here without changing
``env.py``.
"""

import gymnasium as gym
import numpy as np

FULL_GRID_OBSERVATION = "full_grid"
COMPACT_OBSERVATION = "compact"
HYBRID_OBSERVATION = "hybrid"
GRID_CNN_OBSERVATION = "grid_cnn"
SIMPLE_PROGRESS_OBSERVATION = "simple_progress"
BASELINE_OBS_V1 = "baseline_obs_v1"
BASELINE_OBS_V2 = "baseline_obs_v2"
BASELINE_OBS_V3 = "baseline_obs_v3"
BASELINE_OBS_V4 = "baseline_obs_v4"

SPARSE_REWARD = "sparse"
COVERAGE_REWARD = "coverage"
SAFETY_REWARD = "safety"
BASELINE_COVERAGE_REWARD = "baseline_coverage"
BASELINE_REWARD_V1 = "baseline_reward_v1"
BASELINE_REWARD_V2 = "baseline_reward_v2"
BASELINE_REWARD_V3 = "baseline_reward_v3"
BASELINE_REWARD_V4 = "baseline_reward_v4"

OBSERVATION_MODES = (
    FULL_GRID_OBSERVATION,
    COMPACT_OBSERVATION,
    HYBRID_OBSERVATION,
    GRID_CNN_OBSERVATION,
    SIMPLE_PROGRESS_OBSERVATION,
    BASELINE_OBS_V1,
    BASELINE_OBS_V2,
    BASELINE_OBS_V3,
    BASELINE_OBS_V4,
)

REWARD_MODES = (
    SPARSE_REWARD,
    COVERAGE_REWARD,
    SAFETY_REWARD,
    BASELINE_COVERAGE_REWARD,
    BASELINE_REWARD_V1,
    BASELINE_REWARD_V2,
    BASELINE_REWARD_V3,
    BASELINE_REWARD_V4,
)

BLACK = np.asarray((0, 0, 0), dtype=np.uint8)
WHITE = np.asarray((255, 255, 255), dtype=np.uint8)
BROWN = np.asarray((101, 67, 33), dtype=np.uint8)
GREEN = np.asarray((31, 198, 0), dtype=np.uint8)
RED = np.asarray((255, 0, 0), dtype=np.uint8)
LIGHT_RED = np.asarray((255, 127, 127), dtype=np.uint8)

ACTIVE_OBSERVATION_MODE = FULL_GRID_OBSERVATION
ACTIVE_REWARD_MODE = COVERAGE_REWARD
ACTIVE_ENV = None
RUNTIME_TRACKER = {
    "initialized": False,
    "env_id": None,
    "last_agent_pos": 0,
    "position_history": [0],
    "no_position_change_streak": 0,
    "last_steps_remaining": None,
    "last_total_covered_cells": None,
    "last_frontier_distance": None,
}
LAST_RAW_INFO_ID = None
LAST_ENRICHED_INFO: dict | None = None


def configure_runtime(env, observation_mode: str, reward_mode: str) -> None:
    """Bind the active env and selected modes to the custom runtime.

    The immutable environment only passes a raw grid into ``observation()`` and a
    minimal ``info`` dictionary into ``reward()``. This helper gives the custom
    module access to the live env so richer observations and reconstructed reward
    features can be computed externally.
    """
    global ACTIVE_ENV, ACTIVE_OBSERVATION_MODE, ACTIVE_REWARD_MODE
    ACTIVE_ENV = env
    ACTIVE_OBSERVATION_MODE = observation_mode
    ACTIVE_REWARD_MODE = reward_mode
    setattr(env, "observation_mode", observation_mode)
    setattr(env, "reward_mode", reward_mode)
    setattr(env, "observation_space", observation_space(env))
    _initialize_runtime_tracker(env)


def _initialize_runtime_tracker(env: gym.Env | None) -> None:
    """Reset cross-step bookkeeping for the active environment instance."""
    global RUNTIME_TRACKER, LAST_RAW_INFO_ID, LAST_ENRICHED_INFO

    if env is None:
        RUNTIME_TRACKER = {
            "initialized": False,
            "env_id": None,
            "last_agent_pos": 0,
            "position_history": [0],
            "no_position_change_streak": 0,
            "last_steps_remaining": None,
            "last_total_covered_cells": None,
            "last_frontier_distance": None,
        }
    else:
        agent_pos = int(getattr(env, "agent_pos", 0))
        steps_remaining = int(getattr(env, "steps_remaining", 500))
        total_covered_cells = int(getattr(env, "total_covered_cells", 1))
        RUNTIME_TRACKER = {
            "initialized": True,
            "env_id": id(env),
            "last_agent_pos": agent_pos,
            "position_history": [agent_pos],
            "no_position_change_streak": 0,
            "last_steps_remaining": steps_remaining,
            "last_total_covered_cells": total_covered_cells,
            "last_frontier_distance": _nearest_frontier_distance(env),
        }

    LAST_RAW_INFO_ID = None
    LAST_ENRICHED_INFO = None


def _sync_runtime(env: gym.Env | None) -> None:
    """Keep the active env reference synchronized and detect episode resets."""
    global ACTIVE_ENV
    if env is None:
        return

    ACTIVE_ENV = env
    if _runtime_reset_detected(env):
        _initialize_runtime_tracker(env)


def _runtime_reset_detected(env: gym.Env) -> bool:
    """Infer whether the env has started a new episode.

    Because ``env.py`` is immutable, the custom runtime infers resets from the
    externally visible state instead of relying on env-side callbacks.
    """
    if not RUNTIME_TRACKER["initialized"]:
        return True
    if RUNTIME_TRACKER["env_id"] != id(env):
        return True

    current_steps_remaining = int(getattr(env, "steps_remaining", 500))
    current_total_covered = int(getattr(env, "total_covered_cells", 1))
    current_agent_pos = int(getattr(env, "agent_pos", 0))
    last_steps_remaining = RUNTIME_TRACKER["last_steps_remaining"]
    last_total_covered = RUNTIME_TRACKER["last_total_covered_cells"]

    if last_steps_remaining is None:
        return True
    if current_steps_remaining > last_steps_remaining:
        return True
    if current_total_covered < (last_total_covered if last_total_covered is not None else current_total_covered):
        return True
    if current_steps_remaining == 500 and current_total_covered == 1 and current_agent_pos == 0:
        if last_steps_remaining != 500 or RUNTIME_TRACKER["last_agent_pos"] != 0:
            return True

    return False


def _agent_in_enemy_fov(env: gym.Env) -> bool:
    """Return whether the agent is currently visible to any enemy."""
    agent_row = env.agent_pos // env.grid_size
    agent_col = env.agent_pos % env.grid_size
    return any((agent_row, agent_col) in enemy.get_fov_cells() for enemy in env.enemy_list)


def _nearest_frontier_distance(env: gym.Env) -> int | None:
    """Compute the shortest-path distance to the nearest uncovered reachable cell.

    ``BLACK`` and ``RED`` cells both count as frontier because they still need to
    be covered. Walls and enemy tiles are treated as blocked.
    """
    start_row = env.agent_pos // env.grid_size
    start_col = env.agent_pos % env.grid_size

    start_cell = env.grid[start_row, start_col]
    if np.array_equal(start_cell, BLACK) or np.array_equal(start_cell, RED):
        return 0

    visited = np.zeros((env.grid_size, env.grid_size), dtype=bool)
    queue: list[tuple[int, int, int]] = [(start_row, start_col, 0)]
    visited[start_row, start_col] = True
    head = 0

    while head < len(queue):
        row, col, distance = queue[head]
        head += 1

        for delta_row, delta_col in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            next_row = row + delta_row
            next_col = col + delta_col
            if not (0 <= next_row < env.grid_size and 0 <= next_col < env.grid_size):
                continue
            if visited[next_row, next_col]:
                continue

            cell = env.grid[next_row, next_col]
            if np.array_equal(cell, BROWN) or np.array_equal(cell, GREEN):
                continue

            next_distance = distance + 1
            if np.array_equal(cell, BLACK) or np.array_equal(cell, RED):
                return next_distance

            visited[next_row, next_col] = True
            queue.append((next_row, next_col, next_distance))

    return None


def _peek_enriched_info(info: dict) -> dict:
    """Add env-derived fields that can be reconstructed without advancing state."""
    if ACTIVE_ENV is None:
        return dict(info)

    env = ACTIVE_ENV
    _sync_runtime(env)

    enriched = dict(info)
    enriched.setdefault("agent_pos", int(getattr(env, "agent_pos", 0)))
    enriched.setdefault("total_covered_cells", int(getattr(env, "total_covered_cells", 0)))
    enriched.setdefault(
        "coverable_cells",
        int(getattr(env, "coverable_cells", info.get("coverable_cells", 0))),
    )
    enriched.setdefault(
        "cells_remaining",
        int(enriched["coverable_cells"]) - int(enriched["total_covered_cells"]),
    )
    enriched.setdefault("steps_remaining", int(getattr(env, "steps_remaining", info.get("steps_remaining", 0))))
    enriched.setdefault("game_over", bool(getattr(env, "game_over", info.get("game_over", False))))
    enriched.setdefault("in_enemy_fov", _agent_in_enemy_fov(env))
    enriched.setdefault(
        "mission_success",
        bool(enriched["coverable_cells"] == enriched["total_covered_cells"] and not enriched["game_over"]),
    )
    return enriched


def _advance_runtime_tracker(enriched_info: dict) -> None:
    """Commit per-step reconstructed state after reward shaping is computed."""
    global RUNTIME_TRACKER

    agent_pos = int(enriched_info["agent_pos"])
    position_history = list(RUNTIME_TRACKER["position_history"])
    position_history.append(agent_pos)
    if len(position_history) > 6:
        position_history.pop(0)

    RUNTIME_TRACKER["initialized"] = True
    RUNTIME_TRACKER["env_id"] = id(ACTIVE_ENV) if ACTIVE_ENV is not None else None
    RUNTIME_TRACKER["last_agent_pos"] = agent_pos
    RUNTIME_TRACKER["position_history"] = position_history
    RUNTIME_TRACKER["no_position_change_streak"] = int(enriched_info["no_position_change_streak"])
    RUNTIME_TRACKER["last_steps_remaining"] = int(enriched_info["steps_remaining"])
    RUNTIME_TRACKER["last_total_covered_cells"] = int(enriched_info["total_covered_cells"])
    RUNTIME_TRACKER["last_frontier_distance"] = enriched_info.get("frontier_distance")


def _enrich_step_info(info: dict) -> dict:
    """Reconstruct the richer step fields that the original env no longer emits."""
    global LAST_RAW_INFO_ID, LAST_ENRICHED_INFO

    enriched = _peek_enriched_info(info)
    env = ACTIVE_ENV
    if env is None:
        LAST_RAW_INFO_ID = id(info)
        LAST_ENRICHED_INFO = enriched
        return enriched

    previous_agent_pos = int(RUNTIME_TRACKER["last_agent_pos"])
    position_history = RUNTIME_TRACKER["position_history"]
    new_cell_covered = bool(enriched.get("new_cell_covered", False))
    current_agent_pos = int(enriched["agent_pos"])
    no_position_change = current_agent_pos == previous_agent_pos
    no_position_change_streak = (
        int(RUNTIME_TRACKER["no_position_change_streak"]) + 1 if no_position_change else 0
    )
    two_step_oscillation = bool(
        len(position_history) >= 2
        and current_agent_pos == position_history[-2]
        and previous_agent_pos != current_agent_pos
    )
    stationary_without_progress = bool(no_position_change and not new_cell_covered)
    revisited_cell = bool((not no_position_change) and (not new_cell_covered))
    frontier_distance = _nearest_frontier_distance(env)
    previous_frontier_distance = RUNTIME_TRACKER["last_frontier_distance"]

    enriched.update(
        {
            "revisited_cell": bool(enriched.get("revisited_cell", revisited_cell)),
            "no_position_change": bool(enriched.get("no_position_change", no_position_change)),
            "no_position_change_streak": int(
                enriched.get("no_position_change_streak", no_position_change_streak)
            ),
            "two_step_oscillation": bool(
                enriched.get("two_step_oscillation", two_step_oscillation)
            ),
            "in_enemy_fov": bool(enriched.get("in_enemy_fov", _agent_in_enemy_fov(env))),
            "mission_success": bool(
                enriched.get(
                    "mission_success",
                    enriched["coverable_cells"] == enriched["total_covered_cells"] and not enriched["game_over"],
                )
            ),
            "stationary_without_progress": stationary_without_progress,
            "frontier_distance": frontier_distance,
            "previous_frontier_distance": previous_frontier_distance,
        }
    )

    _advance_runtime_tracker(enriched)
    LAST_RAW_INFO_ID = id(info)
    LAST_ENRICHED_INFO = dict(enriched)
    return enriched


def enrich_info(info: dict) -> dict:
    """Return a non-mutating enriched view of a raw env ``info`` dictionary."""
    if LAST_RAW_INFO_ID == id(info) and LAST_ENRICHED_INFO is not None:
        return dict(LAST_ENRICHED_INFO)
    return _peek_enriched_info(info)


def _normalized_agent_position(agent_pos: int, grid_size: int) -> tuple[float, float]:
    """Normalize the flattened agent index into row/column coordinates in [0, 1]."""
    row = agent_pos // grid_size
    col = agent_pos % grid_size
    row_scale = max(grid_size - 1, 1)
    col_scale = max(grid_size - 1, 1)
    return row / row_scale, col / col_scale


def _normalized_grid(env: gym.Env) -> np.ndarray:
    """Return the flattened RGB grid normalized to [0, 1]."""
    return env.grid.astype(np.float32).flatten() / 255.0


def _compact_features(env: gym.Env) -> np.ndarray:
    """Build a minimal progress summary used by the early compact baselines."""
    agent_row, agent_col = _normalized_agent_position(env.agent_pos, env.grid_size)
    coverable_cells = max(env.coverable_cells, 1)
    return np.asarray(
        [
            agent_row,
            agent_col,
            env.total_covered_cells / coverable_cells,
            (coverable_cells - env.total_covered_cells) / coverable_cells,
            env.steps_remaining / 500.0,
            len(env.enemy_list) / float(env.num_cells),
            float(env.game_over),
        ],
        dtype=np.float32,
    )


def _local_action_features(env: gym.Env) -> np.ndarray:
    """Describe immediate move validity, novelty, and local risk for four moves."""
    movement = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    agent_x = env.agent_pos % env.grid_size
    agent_y = env.agent_pos // env.grid_size

    move_validity: list[float] = []
    move_novelty: list[float] = []
    move_risk: list[float] = []

    for delta_y, delta_x in movement:
        y = agent_y + delta_y
        x = agent_x + delta_x

        if 0 <= x < env.grid_size and 0 <= y < env.grid_size:
            cell = env.grid[y, x]
            blocked = int(np.array_equal(cell, BROWN) or np.array_equal(cell, GREEN))
            move_validity.append(1.0 - blocked)
            move_novelty.append(float(np.array_equal(cell, BLACK) or np.array_equal(cell, RED)))
            move_risk.append(float(np.array_equal(cell, RED) or np.array_equal(cell, LIGHT_RED)))
        else:
            move_validity.append(0.0)
            move_novelty.append(0.0)
            move_risk.append(1.0)

    return np.asarray(move_validity + move_novelty + move_risk, dtype=np.float32)


def _simple_progress_features(env: gym.Env) -> np.ndarray:
    """Return the first successful compact baseline observation."""
    agent_row, agent_col = _normalized_agent_position(env.agent_pos, env.grid_size)
    coverable_cells = max(env.coverable_cells, 1)

    movement = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    agent_x = env.agent_pos % env.grid_size
    agent_y = env.agent_pos // env.grid_size

    move_validity: list[float] = []
    move_novelty: list[float] = []
    for delta_y, delta_x in movement:
        y = agent_y + delta_y
        x = agent_x + delta_x
        if 0 <= x < env.grid_size and 0 <= y < env.grid_size:
            cell = env.grid[y, x]
            blocked = float(np.array_equal(cell, BROWN) or np.array_equal(cell, GREEN))
            move_validity.append(1.0 - blocked)
            move_novelty.append(float(np.array_equal(cell, BLACK) or np.array_equal(cell, RED)))
        else:
            move_validity.append(0.0)
            move_novelty.append(0.0)

    return np.asarray(
        [
            agent_row,
            agent_col,
            env.total_covered_cells / coverable_cells,
            env.steps_remaining / 500.0,
            *move_validity,
            *move_novelty,
        ],
        dtype=np.float32,
    )


def _baseline_obs_v2_features(env: gym.Env) -> np.ndarray:
    """Extend the compact baseline with directional unexplored-mass signals."""
    agent_row, agent_col = _normalized_agent_position(env.agent_pos, env.grid_size)
    coverable_cells = max(env.coverable_cells, 1)

    movement = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    agent_x = env.agent_pos % env.grid_size
    agent_y = env.agent_pos // env.grid_size

    move_validity: list[float] = []
    move_novelty: list[float] = []
    for delta_y, delta_x in movement:
        y = agent_y + delta_y
        x = agent_x + delta_x
        if 0 <= x < env.grid_size and 0 <= y < env.grid_size:
            cell = env.grid[y, x]
            blocked = float(np.array_equal(cell, BROWN) or np.array_equal(cell, GREEN))
            move_validity.append(1.0 - blocked)
            move_novelty.append(float(np.array_equal(cell, BLACK) or np.array_equal(cell, RED)))
        else:
            move_validity.append(0.0)
            move_novelty.append(0.0)

    unexplored_mask = np.all(env.grid == BLACK, axis=2) | np.all(env.grid == RED, axis=2)
    total_unexplored = max(int(np.sum(unexplored_mask)), 1)
    unexplored_above = float(np.sum(unexplored_mask[:agent_y, :])) / total_unexplored
    unexplored_below = float(np.sum(unexplored_mask[agent_y + 1 :, :])) / total_unexplored
    unexplored_left = float(np.sum(unexplored_mask[:, :agent_x])) / total_unexplored
    unexplored_right = float(np.sum(unexplored_mask[:, agent_x + 1 :])) / total_unexplored

    return np.asarray(
        [
            agent_row,
            agent_col,
            env.total_covered_cells / coverable_cells,
            env.steps_remaining / 500.0,
            *move_validity,
            *move_novelty,
            unexplored_above,
            unexplored_below,
            unexplored_left,
            unexplored_right,
        ],
        dtype=np.float32,
    )


def _baseline_obs_v3_features(env: gym.Env) -> np.ndarray:
    """Add immediate guard-awareness features to the v2 baseline."""
    base_features = _baseline_obs_v2_features(env)

    movement = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    agent_x = env.agent_pos % env.grid_size
    agent_y = env.agent_pos // env.grid_size

    adjacent_risk: list[float] = []
    for delta_y, delta_x in movement:
        y = agent_y + delta_y
        x = agent_x + delta_x
        if 0 <= x < env.grid_size and 0 <= y < env.grid_size:
            cell = env.grid[y, x]
            adjacent_risk.append(float(np.array_equal(cell, RED) or np.array_equal(cell, LIGHT_RED)))
        else:
            adjacent_risk.append(1.0)

    current_tile_risk = float(np.array_equal(env.grid[agent_y, agent_x], RED) or np.array_equal(env.grid[agent_y, agent_x], LIGHT_RED))

    return np.concatenate(
        [
            base_features,
            np.asarray(adjacent_risk + [current_tile_risk], dtype=np.float32),
        ]
    ).astype(np.float32)


def _is_visible_for_enemy(env: gym.Env, row: int, col: int) -> bool:
    """Check whether an enemy FOV ray can traverse a given cell."""
    if row < 0 or col < 0 or row >= env.grid_size or col >= env.grid_size:
        return False
    cell = env.grid[row, col]
    return not (np.array_equal(cell, BROWN) or np.array_equal(cell, GREEN))


def _forecast_enemy_fov_cells(env: gym.Env, enemy, steps_ahead: int) -> set[tuple[int, int]]:
    """Forecast the cells a given enemy will observe after future rotations."""
    orientation = (enemy.orientation + steps_ahead) % 4
    fov_cells: set[tuple[int, int]] = set()

    for distance in range(1, env.enemy_fov_distance + 1):
        if orientation == 0:  # LEFT
            row, col = enemy.y, enemy.x - distance
        elif orientation == 1:  # DOWN
            row, col = enemy.y + distance, enemy.x
        elif orientation == 2:  # RIGHT
            row, col = enemy.y, enemy.x + distance
        else:  # UP
            row, col = enemy.y - distance, enemy.x

        if _is_visible_for_enemy(env, row, col):
            fov_cells.add((row, col))
        else:
            break

    return fov_cells


def _baseline_obs_v4_features(env: gym.Env) -> np.ndarray:
    """Add short-horizon enemy FOV forecasts for precise timing decisions."""
    base_features = _baseline_obs_v3_features(env)

    movement = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    agent_x = env.agent_pos % env.grid_size
    agent_y = env.agent_pos // env.grid_size

    forecast_features: list[float] = []
    for steps_ahead in range(1, 5):
        forecasted_fov: set[tuple[int, int]] = set()
        for enemy in env.enemy_list:
            forecasted_fov.update(_forecast_enemy_fov_cells(env, enemy, steps_ahead))

        for delta_y, delta_x in movement:
            row = agent_y + delta_y
            col = agent_x + delta_x
            if 0 <= col < env.grid_size and 0 <= row < env.grid_size:
                forecast_features.append(float((row, col) in forecasted_fov))
            else:
                forecast_features.append(1.0)

    return np.concatenate(
        [
            base_features,
            np.asarray(forecast_features, dtype=np.float32),
        ]
    ).astype(np.float32)


def _no_movement_penalty(streak: int, reward_mode: str) -> float:
    """Return escalating stagnation penalties for the legacy reward families."""
    if streak <= 0:
        return 0.0

    if reward_mode == COVERAGE_REWARD:
        schedule = [0.0, 0.02, 0.04, 0.07, 0.10]
        return schedule[min(streak, 4)]

    if reward_mode == SAFETY_REWARD:
        schedule = [0.0, 0.03, 0.05, 0.08, 0.12]
        return schedule[min(streak, 4)]

    return 0.0


def _frontier_progress_reward(
    frontier_distance: int | None,
    previous_frontier_distance: int | None,
    *,
    toward_reward: float,
    away_penalty: float,
    flat_penalty: float,
) -> float:
    """Reward moving toward the nearest frontier and penalize drifting away."""
    if frontier_distance is None or previous_frontier_distance is None:
        return 0.0
    if frontier_distance < previous_frontier_distance:
        return toward_reward
    if frontier_distance > previous_frontier_distance:
        return -away_penalty
    return -flat_penalty


def observation_space(env: gym.Env) -> gym.spaces.Space:
    """Return the Gymnasium observation space for the selected observation mode."""
    observation_mode = getattr(env, "observation_mode", ACTIVE_OBSERVATION_MODE)

    if observation_mode == FULL_GRID_OBSERVATION:
        return gym.spaces.Box(
            low=0,
            high=255,
            shape=(env.grid_size * env.grid_size * 3,),
            dtype=np.uint8,
        )

    if observation_mode == COMPACT_OBSERVATION:
        return gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(7,),
            dtype=np.float32,
        )

    if observation_mode == HYBRID_OBSERVATION:
        return gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(env.grid_size * env.grid_size * 3 + 19,),
            dtype=np.float32,
        )

    if observation_mode == GRID_CNN_OBSERVATION:
        return gym.spaces.Box(
            low=0,
            high=255,
            shape=(3, env.grid_size, env.grid_size),
            dtype=np.uint8,
        )

    if observation_mode == SIMPLE_PROGRESS_OBSERVATION:
        return gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(12,),
            dtype=np.float32,
        )

    if observation_mode == BASELINE_OBS_V1:
        return gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(12,),
            dtype=np.float32,
        )

    if observation_mode == BASELINE_OBS_V2:
        return gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(16,),
            dtype=np.float32,
        )

    if observation_mode == BASELINE_OBS_V3:
        return gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(21,),
            dtype=np.float32,
        )

    if observation_mode == BASELINE_OBS_V4:
        return gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(37,),
            dtype=np.float32,
        )

    raise ValueError(f"Unsupported observation mode: {observation_mode}")


def observation(env_or_grid):
    """Build the observation for the current state.

    ``env.py`` passes a raw grid here. Non-default modes therefore recover the
    live env from the configured runtime and compute features from that env.
    """
    if isinstance(env_or_grid, np.ndarray):
        if ACTIVE_ENV is None or ACTIVE_OBSERVATION_MODE == FULL_GRID_OBSERVATION:
            return env_or_grid.flatten()
        _sync_runtime(ACTIVE_ENV)
        env = ACTIVE_ENV
    else:
        env = env_or_grid
        _sync_runtime(env)

    observation_mode = getattr(env, "observation_mode", ACTIVE_OBSERVATION_MODE)

    if observation_mode == FULL_GRID_OBSERVATION:
        return env.grid.flatten()

    if observation_mode == COMPACT_OBSERVATION:
        return _compact_features(env)

    if observation_mode == HYBRID_OBSERVATION:
        return np.concatenate(
            [
                _normalized_grid(env),
                _compact_features(env),
                _local_action_features(env),
            ]
        ).astype(np.float32)

    if observation_mode == GRID_CNN_OBSERVATION:
        return np.transpose(env.grid, (2, 0, 1)).astype(np.uint8)

    if observation_mode == SIMPLE_PROGRESS_OBSERVATION:
        return _simple_progress_features(env)

    if observation_mode == BASELINE_OBS_V1:
        return _simple_progress_features(env)

    if observation_mode == BASELINE_OBS_V2:
        return _baseline_obs_v2_features(env)

    if observation_mode == BASELINE_OBS_V3:
        return _baseline_obs_v3_features(env)

    if observation_mode == BASELINE_OBS_V4:
        return _baseline_obs_v4_features(env)

    raise ValueError(f"Unsupported observation mode: {observation_mode}")


def reward(info: dict, reward_mode: str = None) -> float:
    """Calculate the step reward for the selected reward mode.

    The immutable env provides only a minimal ``info`` dictionary, so this
    function first reconstructs the richer step-level fields required by the
    later baseline rewards before applying the chosen shaping logic.
    """
    reward_mode = reward_mode or ACTIVE_REWARD_MODE

    enriched = _enrich_step_info(info)
    cells_remaining = enriched.get("cells_remaining", 0)
    coverable_cells = max(enriched.get("coverable_cells", 1), 1)
    steps_remaining = enriched.get("steps_remaining", 0)
    new_cell_covered = enriched.get("new_cell_covered", False)
    game_over = enriched.get("game_over", False)
    stayed_still = enriched.get("stayed_still", False)
    move_blocked = enriched.get("move_blocked", False)
    revisited_cell = enriched.get("revisited_cell", False)
    no_position_change = enriched.get("no_position_change", False)
    no_position_change_streak = enriched.get("no_position_change_streak", 0)
    two_step_oscillation = enriched.get("two_step_oscillation", False)
    in_enemy_fov = enriched.get("in_enemy_fov", False)
    mission_success = enriched.get("mission_success", False)
    stationary_without_progress = enriched.get("stationary_without_progress", False)
    frontier_distance = enriched.get("frontier_distance")
    previous_frontier_distance = enriched.get("previous_frontier_distance")

    if reward_mode == SPARSE_REWARD:
        if game_over:
            return -1.0
        if mission_success:
            return 1.0
        return 0.0

    if reward_mode in {BASELINE_COVERAGE_REWARD, BASELINE_REWARD_V1}:
        value = -0.01
        if new_cell_covered:
            value += 1.0
        if move_blocked or stayed_still or stationary_without_progress:
            value -= 0.05
        if mission_success:
            value += 50.0
        if two_step_oscillation and not new_cell_covered:
            value -= 0.2
        return value

    if reward_mode == BASELINE_REWARD_V2:
        value = -0.01
        if new_cell_covered:
            value += 1.0
        if move_blocked or stayed_still or stationary_without_progress:
            value -= 0.05
        if mission_success:
            value += 50.0
        if two_step_oscillation and not new_cell_covered:
            value -= 0.2
        if in_enemy_fov:
            value -= 0.1
        if game_over:
            value -= 4.0
        return value

    if reward_mode == BASELINE_REWARD_V3:
        value = -0.03
        if new_cell_covered:
            value += 1.0
        if move_blocked or stayed_still or stationary_without_progress:
            value -= 0.05
        if mission_success:
            value += 50.0
        if two_step_oscillation and not new_cell_covered:
            value -= 0.2
        if in_enemy_fov:
            value -= 0.1
        if game_over:
            value -= 4.0
        if cells_remaining <= 10:
            if new_cell_covered:
                value += 0.2
            else:
                value -= 0.1
        if steps_remaining <= 0 and not mission_success:
            value -= 2.0 * cells_remaining
        return value

    if reward_mode == BASELINE_REWARD_V4:
        value = -0.03
        if new_cell_covered:
            value += 1.0
        if move_blocked or stayed_still or stationary_without_progress:
            value -= 0.05
        if mission_success:
            value += 50.0
        if two_step_oscillation and not new_cell_covered:
            value -= 0.2
        if in_enemy_fov:
            value -= 0.1
        if game_over:
            value -= 4.0

        if not new_cell_covered:
            value += _frontier_progress_reward(
                frontier_distance,
                previous_frontier_distance,
                toward_reward=0.02,
                away_penalty=0.02,
                flat_penalty=0.0,
            )

        if cells_remaining <= 10:
            if new_cell_covered:
                value += 0.2
            else:
                value -= 0.08
                value += _frontier_progress_reward(
                    frontier_distance,
                    previous_frontier_distance,
                    toward_reward=0.12,
                    away_penalty=0.12,
                    flat_penalty=0.04,
                )

        if steps_remaining <= 0 and not mission_success:
            value -= 2.0 * cells_remaining
        return value

    if reward_mode == COVERAGE_REWARD:
        cover_ratio = 1.0 - (cells_remaining / coverable_cells)
        value = -0.01
        if new_cell_covered:
            value += 1.0
            value += 0.1 * cover_ratio
        if revisited_cell:
            value -= 0.03
        if no_position_change:
            value -= _no_movement_penalty(no_position_change_streak, reward_mode)
        if two_step_oscillation:
            value -= 0.08
        if stayed_still:
            value -= 0.02
        if move_blocked:
            value -= 0.08
        if in_enemy_fov:
            value -= 0.2
        if game_over:
            value -= 2.0
        if mission_success:
            value += 25.0
        return value

    if reward_mode == SAFETY_REWARD:
        cover_ratio = 1.0 - (cells_remaining / coverable_cells)
        value = -0.01
        if new_cell_covered:
            value += 0.9
            value += 0.1 * cover_ratio
        if revisited_cell:
            value -= 0.04
        if no_position_change:
            value -= _no_movement_penalty(no_position_change_streak, reward_mode)
        if two_step_oscillation:
            value -= 0.10
        if stayed_still:
            value -= 0.02
        if move_blocked:
            value -= 0.10
        if in_enemy_fov:
            value -= 0.35
        if game_over:
            value -= 2.5
        if mission_success:
            value += 24.0
        if cells_remaining == 0:
            value += 0.5
        return value

    raise ValueError(f"Unsupported reward mode: {reward_mode}")
