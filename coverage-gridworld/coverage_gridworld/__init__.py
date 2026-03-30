"""Register the predefined Coverage Gridworld maps used by the project.

The package exposes ``CoverageGridworld`` for Gymnasium entry points and
registers the fixed map layouts used in training, evaluation, and playback.
Some historical custom maps are kept registered even if they are not part of
the current validated training sets.
"""

from gymnasium.envs.registration import register
from coverage_gridworld.env import CoverageGridworld  # noqa: F401  # Re-exported for Gym registration.

register(
    id="standard",
    entry_point="coverage_gridworld:CoverageGridworld"
)

register(
    id="just_go",   # very easy difficulty
    entry_point="coverage_gridworld:CoverageGridworld",
    kwargs={
        "predefined_map": [
            [3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ]
    }
)

register(
    id="safe",   # easy difficulty
    entry_point="coverage_gridworld:CoverageGridworld",
    kwargs={
        "predefined_map": [
            [3, 0, 0, 2, 0, 2, 0, 0, 0, 0],
            [0, 2, 0, 0, 0, 2, 0, 0, 2, 0],
            [0, 2, 0, 2, 2, 2, 2, 2, 2, 0],
            [0, 2, 0, 0, 0, 2, 0, 0, 0, 0],
            [0, 2, 0, 2, 0, 2, 0, 0, 2, 0],
            [0, 2, 0, 0, 0, 0, 0, 2, 0, 0],
            [0, 2, 2, 2, 0, 0, 0, 2, 0, 0],
            [0, 0, 0, 0, 0, 2, 0, 0, 2, 0],
            [0, 2, 0, 2, 0, 2, 2, 0, 0, 0],
            [0, 0, 0, 0, 0, 2, 0, 0, 0, 0]
        ]
    }
)

register(
    id="maze",   # medium difficulty
    entry_point="coverage_gridworld:CoverageGridworld",
    kwargs={
        "predefined_map": [
            [3, 2, 0, 0, 0, 0, 2, 0, 0, 0],
            [0, 2, 0, 2, 2, 0, 2, 0, 2, 2],
            [0, 2, 0, 2, 0, 0, 2, 0, 0, 0],
            [0, 2, 0, 2, 0, 2, 2, 2, 2, 0],
            [0, 2, 0, 2, 0, 0, 2, 0, 0, 0],
            [0, 2, 0, 2, 2, 0, 2, 0, 2, 2],
            [0, 2, 0, 2, 0, 0, 2, 0, 0, 0],
            [0, 2, 0, 2, 0, 2, 2, 2, 2, 0],
            [0, 2, 0, 2, 0, 4, 2, 4, 0, 0],
            [0, 0, 0, 2, 0, 0, 0, 0, 0, 0]
        ]
    }
)

register(
    id="custom_challenge",   # hard difficulty
    entry_point="coverage_gridworld:CoverageGridworld",
    kwargs={
        "predefined_map": [
            [3, 0, 2, 0, 0, 0, 0, 2, 0, 0],
            [0, 0, 2, 0, 0, 0, 0, 0, 0, 4],
            [0, 4, 2, 2, 0, 2, 0, 2, 0, 0],
            [0, 2, 0, 0, 0, 0, 0, 2, 0, 0],
            [0, 4, 2, 0, 2, 0, 0, 2, 0, 0],
            [0, 0, 2, 0, 2, 0, 0, 0, 0, 0],
            [0, 0, 2, 0, 0, 0, 0, 0, 2, 0],
            [0, 0, 2, 0, 0, 0, 0, 2, 0, 0],
            [0, 0, 0, 0, 4, 0, 4, 2, 0, 0],
            [2, 0, 0, 0, 0, 0, 0, 2, 0, 4]
        ]
    }
)

register(
    id="timing_corridor",   # timing-focused: narrow corridors with sequential guard timing
    entry_point="coverage_gridworld:CoverageGridworld",
    kwargs={
        "predefined_map": [
            [3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 2, 2, 2, 0, 2, 2, 2, 2, 0],
            [0, 0, 0, 2, 0, 2, 0, 0, 0, 0],
            [0, 2, 0, 2, 0, 2, 0, 2, 2, 0],
            [0, 2, 0, 0, 0, 0, 0, 2, 4, 0],
            [0, 2, 2, 2, 2, 2, 0, 2, 0, 0],
            [0, 0, 0, 0, 0, 2, 0, 2, 0, 0],
            [2, 2, 2, 2, 0, 2, 0, 2, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 2, 0, 4],
            [0, 2, 2, 2, 2, 2, 0, 0, 0, 0]
        ]
    }
)

# Historical custom map retained for reference; not part of the validated large training sets.
register(
    id="pocket_patrol",   # cleanup-focused: small pockets near guard influence
    entry_point="coverage_gridworld:CoverageGridworld",
    kwargs={
        "predefined_map": [
            [3, 0, 0, 0, 2, 0, 0, 0, 0, 0],
            [0, 2, 2, 0, 2, 0, 2, 2, 2, 0],
            [0, 0, 0, 0, 2, 0, 0, 0, 2, 0],
            [0, 2, 0, 2, 2, 2, 2, 0, 2, 0],
            [0, 2, 0, 0, 0, 4, 2, 0, 0, 0],
            [0, 2, 2, 2, 0, 0, 2, 2, 2, 0],
            [0, 0, 0, 2, 0, 0, 0, 0, 2, 0],
            [0, 2, 0, 2, 2, 2, 2, 0, 2, 0],
            [0, 2, 0, 0, 0, 0, 0, 0, 2, 4],
            [0, 0, 0, 2, 2, 2, 2, 0, 0, 0]
        ]
    }
)

# Historical custom map retained for reference; not part of the validated large training sets.
register(
    id="crossroads_patrol",   # routing-focused: open spaces with central guard pressure
    entry_point="coverage_gridworld:CoverageGridworld",
    kwargs={
        "predefined_map": [
            [3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 2, 2, 0, 0, 0, 0, 2, 2, 0],
            [0, 2, 0, 0, 2, 2, 0, 0, 2, 0],
            [0, 0, 0, 0, 2, 2, 0, 0, 0, 0],
            [0, 0, 2, 2, 4, 0, 0, 2, 0, 0],
            [0, 0, 2, 0, 0, 4, 2, 2, 0, 0],
            [0, 0, 0, 0, 2, 2, 0, 0, 0, 0],
            [0, 2, 0, 0, 2, 2, 0, 0, 2, 0],
            [0, 2, 2, 0, 0, 0, 0, 2, 2, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ]
    }
)

register(
    id="staggered_escape",   # sequencing-focused: chained guard passages with recovery space
    entry_point="coverage_gridworld:CoverageGridworld",
    kwargs={
        "predefined_map": [
            [3, 0, 2, 0, 0, 0, 0, 2, 0, 0],
            [0, 0, 2, 0, 2, 2, 0, 2, 0, 4],
            [0, 0, 2, 0, 0, 0, 0, 2, 0, 0],
            [2, 0, 2, 2, 2, 0, 2, 2, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 2, 2, 2, 0, 2, 2, 2, 2, 0],
            [0, 0, 0, 2, 0, 0, 0, 2, 0, 0],
            [0, 2, 0, 2, 2, 2, 0, 2, 0, 4],
            [0, 2, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 2, 2, 2, 2, 2, 0, 0]
        ]
    }
)

register(
    id="patrol_weave",   # routing-focused: alternating lanes with staggered guard pressure
    entry_point="coverage_gridworld:CoverageGridworld",
    kwargs={
        "predefined_map": [
            [3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 2, 2, 0, 2, 0, 2, 0, 2, 0],
            [0, 0, 0, 0, 2, 0, 0, 0, 2, 0],
            [0, 2, 0, 2, 2, 0, 2, 0, 2, 0],
            [0, 2, 0, 0, 0, 0, 2, 0, 4, 0],
            [0, 2, 2, 2, 0, 2, 2, 0, 2, 0],
            [0, 0, 0, 2, 0, 0, 0, 0, 2, 0],
            [2, 2, 0, 2, 2, 2, 0, 2, 2, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 4],
            [0, 2, 2, 2, 2, 2, 2, 2, 0, 0]
        ]
    }
)

register(
    id="enemy_spine",   # timing-focused: central vertical route with exposed branches
    entry_point="coverage_gridworld:CoverageGridworld",
    kwargs={
        "predefined_map": [
            [3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 2, 2, 2, 0, 2, 2, 2, 2, 0],
            [0, 0, 0, 2, 0, 2, 0, 0, 0, 0],
            [0, 2, 0, 2, 0, 2, 0, 2, 2, 0],
            [0, 2, 0, 0, 0, 0, 0, 0, 4, 0],
            [0, 2, 2, 2, 0, 2, 2, 0, 2, 0],
            [0, 0, 0, 2, 0, 2, 0, 0, 2, 0],
            [2, 2, 0, 2, 0, 2, 0, 2, 2, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 4],
            [0, 2, 2, 2, 2, 2, 2, 2, 0, 0]
        ]
    }
)

register(
    id="sidepass_patrol",   # cleanup-focused: side corridors with a guarded lower branch
    entry_point="coverage_gridworld:CoverageGridworld",
    kwargs={
        "predefined_map": [
            [3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 2, 2, 2, 0, 2, 2, 2, 0, 0],
            [0, 0, 0, 2, 0, 2, 0, 0, 0, 0],
            [0, 2, 0, 2, 0, 2, 0, 2, 2, 0],
            [0, 2, 0, 0, 0, 0, 0, 2, 4, 0],
            [0, 2, 2, 2, 0, 2, 0, 2, 0, 0],
            [0, 0, 0, 2, 0, 2, 0, 0, 0, 0],
            [2, 2, 0, 2, 0, 2, 2, 2, 2, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 4],
            [0, 2, 2, 2, 2, 2, 0, 2, 0, 0]
        ]
    }
)

register(
    id="triple_patrol",   # enemy-dense: three patrol anchors across a connected sweep map
    entry_point="coverage_gridworld:CoverageGridworld",
    kwargs={
        "predefined_map": [
            [3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 2, 2, 0, 2, 0, 2, 0, 2, 0],
            [0, 0, 0, 0, 2, 0, 0, 0, 2, 4],
            [0, 2, 0, 2, 2, 0, 2, 0, 2, 0],
            [0, 2, 0, 0, 0, 0, 2, 0, 0, 0],
            [0, 2, 2, 2, 0, 2, 2, 2, 0, 0],
            [0, 0, 0, 2, 0, 0, 0, 2, 0, 4],
            [2, 2, 0, 2, 2, 2, 0, 2, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 4],
            [0, 2, 2, 2, 2, 2, 2, 2, 0, 0]
        ]
    }
)

register(
    id="pressure_spokes",   # enemy-dense: connected hub-and-spoke layout with three guard anchors
    entry_point="coverage_gridworld:CoverageGridworld",
    kwargs={
        "predefined_map": [
            [3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 2, 2, 0, 2, 0, 2, 0, 2, 0],
            [0, 0, 0, 0, 2, 0, 0, 0, 2, 4],
            [0, 2, 0, 2, 2, 0, 2, 0, 2, 0],
            [0, 2, 0, 0, 0, 0, 2, 0, 0, 0],
            [0, 2, 2, 2, 0, 2, 2, 2, 0, 0],
            [0, 0, 0, 2, 0, 0, 0, 2, 0, 4],
            [2, 2, 0, 2, 2, 2, 0, 2, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 4],
            [0, 2, 2, 2, 2, 2, 2, 2, 0, 0]
        ]
    }
)

register(
    id="chokepoint",   # hard difficulty
    entry_point="coverage_gridworld:CoverageGridworld",
    kwargs={
        "predefined_map": [
            [3, 0, 2, 0, 0, 0, 0, 2, 0, 0],
            [0, 0, 2, 0, 0, 0, 0, 0, 0, 4],
            [0, 0, 2, 0, 0, 0, 0, 2, 0, 0],
            [0, 0, 2, 0, 0, 0, 0, 2, 0, 0],
            [0, 4, 2, 0, 0, 0, 0, 2, 0, 0],
            [0, 0, 2, 0, 0, 0, 0, 2, 0, 0],
            [0, 0, 2, 0, 0, 0, 0, 2, 0, 0],
            [0, 0, 2, 0, 0, 0, 0, 2, 0, 0],
            [0, 0, 0, 0, 4, 0, 4, 2, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 2, 0, 0]
        ]
    }
)

register(
    id="sneaky_enemies",   # very hard difficulty
    entry_point="coverage_gridworld:CoverageGridworld",
    kwargs={
        "predefined_map": [
            [3, 0, 0, 0, 0, 0, 0, 4, 0, 0],
            [0, 2, 0, 2, 0, 0, 2, 0, 2, 0],
            [0, 0, 0, 0, 4, 0, 0, 0, 0, 0],
            [0, 2, 0, 2, 0, 0, 2, 0, 2, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 2, 0, 2, 0, 0, 2, 0, 2, 0],
            [4, 0, 0, 0, 0, 0, 0, 0, 0, 4],
            [0, 2, 0, 2, 0, 0, 2, 0, 2, 0],
            [0, 0, 0, 0, 0, 4, 0, 0, 0, 0],
            [0, 2, 0, 2, 0, 0, 2, 0, 2, 0]
        ]
    }
)

# To create a predefined map, just add walls and enemies. The agent always starts in the top-left corner.
# The enemy's orientation is randomly defined and the cells under surveillance will be spawned automatically
