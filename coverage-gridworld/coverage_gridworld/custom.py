import gymnasium as gym
import numpy as np

"""
Feel free to modify the functions below and experiment with different environment configurations.
"""

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

BLACK = np.asarray((0, 0, 0), dtype=np.uint8)
WHITE = np.asarray((255, 255, 255), dtype=np.uint8)
BROWN = np.asarray((101, 67, 33), dtype=np.uint8)
GREEN = np.asarray((31, 198, 0), dtype=np.uint8)
RED = np.asarray((255, 0, 0), dtype=np.uint8)
LIGHT_RED = np.asarray((255, 127, 127), dtype=np.uint8)


def _normalized_agent_position(agent_pos: int, grid_size: int) -> tuple[float, float]:
    row = agent_pos // grid_size
    col = agent_pos % grid_size
    row_scale = max(grid_size - 1, 1)
    col_scale = max(grid_size - 1, 1)
    return row / row_scale, col / col_scale


def _normalized_grid(env: gym.Env) -> np.ndarray:
    return env.grid.astype(np.float32).flatten() / 255.0


def _compact_features(env: gym.Env) -> np.ndarray:
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
    if row < 0 or col < 0 or row >= env.grid_size or col >= env.grid_size:
        return False
    cell = env.grid[row, col]
    return not (np.array_equal(cell, BROWN) or np.array_equal(cell, GREEN))


def _forecast_enemy_fov_cells(env: gym.Env, enemy, steps_ahead: int) -> set[tuple[int, int]]:
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
    if streak <= 0:
        return 0.0

    if reward_mode == COVERAGE_REWARD:
        schedule = [0.0, 0.02, 0.04, 0.07, 0.10]
        return schedule[min(streak, 4)]

    if reward_mode == SAFETY_REWARD:
        schedule = [0.0, 0.03, 0.05, 0.08, 0.12]
        return schedule[min(streak, 4)]

    return 0.0


def observation_space(env: gym.Env) -> gym.spaces.Space:
    """
    Observation space from Gymnasium (https://gymnasium.farama.org/api/spaces/)
    """
    observation_mode = getattr(env, "observation_mode", FULL_GRID_OBSERVATION)

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


def observation(env: gym.Env):
    """
    Function that returns the observation for the current state of the environment.
    """
    observation_mode = getattr(env, "observation_mode", FULL_GRID_OBSERVATION)

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


def reward(info: dict, reward_mode: str = COVERAGE_REWARD) -> float:
    """
    Function to calculate the reward for the current step based on the state information.

    The info dictionary has the following keys:
    - enemies (list): list of `Enemy` objects. Each Enemy has the following attributes:
        - x (int): column index,
        - y (int): row index,
        - orientation (int): orientation of the agent (LEFT = 0, DOWN = 1, RIGHT = 2, UP = 3),
        - fov_cells (list): list of integer tuples indicating the coordinates of cells currently observed by the agent,
    - agent_pos (int): agent position considering the flattened grid (e.g. cell `(2, 3)` corresponds to position `23`),
    - total_covered_cells (int): how many cells have been covered by the agent so far,
    - cells_remaining (int): how many cells are left to be visited in the current map layout,
    - coverable_cells (int): how many cells can be covered in the current map layout,
    - steps_remaining (int): steps remaining in the episode,
    - new_cell_covered (bool): if a cell previously uncovered was covered on this step,
    - game_over (bool): if the game was terminated because the player was seen by an enemy or not,
    - stayed_still (bool): if the chosen action was STAY,
    - move_blocked (bool): if the chosen move hit a wall, enemy, or map boundary,
    - in_enemy_fov (bool): if the agent ended the step inside an enemy field of view,
    - mission_success (bool): if the map was fully covered without losing.
    """
    cells_remaining = info["cells_remaining"]
    new_cell_covered = info["new_cell_covered"]
    game_over = info["game_over"]
    stayed_still = info["stayed_still"]
    move_blocked = info["move_blocked"]
    revisited_cell = info["revisited_cell"]
    no_position_change = info["no_position_change"]
    no_position_change_streak = info["no_position_change_streak"]
    two_step_oscillation = info["two_step_oscillation"]
    in_enemy_fov = info["in_enemy_fov"]
    mission_success = info["mission_success"]

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
        if move_blocked or stayed_still:
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
        if move_blocked or stayed_still:
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
        value = -0.01
        if new_cell_covered:
            value += 1.0
        if move_blocked or stayed_still:
            value -= 0.05
        if mission_success:
            value += 50.0
        if two_step_oscillation and not new_cell_covered:
            value -= 0.2
        if in_enemy_fov:
            value -= 0.1
        if game_over:
            value -= 4.0
        if cells_remaining <= 5 and not new_cell_covered:
            value -= 0.03
        if info["steps_remaining"] <= 0 and not mission_success:
            value -= 2.0 * cells_remaining
        return value

    if reward_mode == COVERAGE_REWARD:
        cover_ratio = 1.0 - (cells_remaining / max(info["coverable_cells"], 1))
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
        cover_ratio = 1.0 - (cells_remaining / max(info["coverable_cells"], 1))
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
