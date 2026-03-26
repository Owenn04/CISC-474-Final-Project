import gymnasium as gym
import numpy as np

"""
Feel free to modify the functions below and experiment with different environment configurations.
"""

FULL_GRID_OBSERVATION = "full_grid"
COMPACT_OBSERVATION = "compact"

SPARSE_REWARD = "sparse"
COVERAGE_REWARD = "coverage"
SAFETY_REWARD = "safety"


def _normalized_agent_position(agent_pos: int, grid_size: int) -> tuple[float, float]:
    row = agent_pos // grid_size
    col = agent_pos % grid_size
    row_scale = max(grid_size - 1, 1)
    col_scale = max(grid_size - 1, 1)
    return row / row_scale, col / col_scale


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

    raise ValueError(f"Unsupported observation mode: {observation_mode}")


def observation(env: gym.Env):
    """
    Function that returns the observation for the current state of the environment.
    """
    observation_mode = getattr(env, "observation_mode", FULL_GRID_OBSERVATION)

    if observation_mode == FULL_GRID_OBSERVATION:
        return env.grid.flatten()

    if observation_mode == COMPACT_OBSERVATION:
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
    in_enemy_fov = info["in_enemy_fov"]
    mission_success = info["mission_success"]

    if reward_mode == SPARSE_REWARD:
        if game_over:
            return -1.0
        if mission_success:
            return 1.0
        return 0.0

    if reward_mode == COVERAGE_REWARD:
        value = -0.01
        if new_cell_covered:
            value += 1.0
        if stayed_still:
            value -= 0.05
        if move_blocked:
            value -= 0.1
        if game_over:
            value -= 2.0
        if mission_success:
            value += 3.0
        return value

    if reward_mode == SAFETY_REWARD:
        value = -0.02
        if new_cell_covered:
            value += 0.75
        if stayed_still:
            value -= 0.05
        if move_blocked:
            value -= 0.1
        if in_enemy_fov:
            value -= 0.25
        if game_over:
            value -= 3.0
        if mission_success:
            value += 4.0
        if cells_remaining == 0:
            value += 0.5
        return value

    raise ValueError(f"Unsupported reward mode: {reward_mode}")
