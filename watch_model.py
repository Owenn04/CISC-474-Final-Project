from __future__ import annotations

"""Render saved Coverage Gridworld models for qualitative inspection.

The environment already knows how to render itself; this script provides the
playback harness that loads a trained SB3 model, configures the runtime modes,
and prints concise episode summaries while the map is being shown.
"""

import argparse
import re
import time
from pathlib import Path

import coverage_gridworld  # noqa: F401  # Required for Gymnasium env registration
import gymnasium as gym
from coverage_gridworld import custom as custom_runtime
from stable_baselines3 import DQN, PPO


def _build_mode_pattern(values: tuple[str, ...]) -> str:
    """Return a regex alternation that safely matches the given mode names."""

    return "|".join(re.escape(value) for value in values)


MODEL_PATTERN = re.compile(
    rf"^(?P<algorithm>ppo|dqn)_(?P<env_id>.+)_(?P<observation_mode>{_build_mode_pattern(custom_runtime.OBSERVATION_MODES)})_(?P<reward_mode>{_build_mode_pattern(custom_runtime.REWARD_MODES)})_(?P<timesteps>\d+)\.zip$"
)

MODEL_CLASSES = {
    "ppo": PPO,
    "dqn": DQN,
}

MAP_CHOICES = [
    "standard",
    "just_go",
    "safe",
    "maze",
    "custom_challenge",
    "timing_corridor",
    "staggered_escape",
    "patrol_weave",
    "enemy_spine",
    "sidepass_patrol",
    "triple_patrol",
    "pressure_spokes",
    "chokepoint",
    "sneaky_enemies",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for model playback."""

    parser = argparse.ArgumentParser(
        description="Render a saved model playing Coverage Gridworld on a selected map."
    )
    parser.add_argument("--model", required=True, help="Path to a saved SB3 .zip model.")
    parser.add_argument(
        "--map",
        default=None,
        choices=MAP_CHOICES,
        help="Environment id to watch. Defaults to the env id encoded in the model filename.",
    )
    parser.add_argument("--episodes", type=int, default=3, help="Number of rendered episodes to watch.")
    parser.add_argument("--seed", type=int, default=42, help="Base seed used for resets.")
    parser.add_argument("--delay", type=float, default=0.15, help="Optional sleep in seconds after each step.")
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample actions stochastically instead of using deterministic predictions.",
    )
    parser.add_argument(
        "--show-status",
        action="store_true",
        help="Print environment game-status messages such as VICTORY and GAME OVER.",
    )
    return parser.parse_args()


def parse_model_metadata(model_path: Path) -> dict[str, str]:
    """Extract algorithm and runtime modes from the saved model filename."""

    match = MODEL_PATTERN.match(model_path.name)
    if not match:
        raise ValueError(
            "Model filename does not match the expected pattern. "
            "Use the existing training filename format such as "
            "'ppo_standard_full_grid_coverage_50000.zip'."
        )
    return match.groupdict()


def main() -> None:
    """Load a saved model, roll it out, and print episode-level results."""

    args = parse_args()
    model_path = Path(args.model)
    metadata = parse_model_metadata(model_path)

    algorithm = metadata["algorithm"]
    env_id = args.map or metadata["env_id"]
    observation_mode = metadata["observation_mode"]
    reward_mode = metadata["reward_mode"]

    model = MODEL_CLASSES[algorithm].load(model_path)
    env = gym.make(
        env_id,
        render_mode="human",
        predefined_map_list=None,
        activate_game_status=args.show_status,
        observation_mode=observation_mode,
        reward_mode=reward_mode,
    )
    custom_runtime.configure_runtime(env.unwrapped, observation_mode, reward_mode)
    env.unwrapped.observation_space = custom_runtime.observation_space(env.unwrapped)
    env.reset(seed=args.seed)

    try:
        for episode_index in range(args.episodes):
            obs, _ = env.reset(seed=args.seed + episode_index)
            done = False
            total_reward = 0.0
            steps = 0
            last_info: dict | None = None

            while not done:
                action, _ = model.predict(obs, deterministic=not args.stochastic)
                obs, reward, terminated, truncated, info = env.step(action)
                info = custom_runtime.enrich_info(info)
                total_reward += float(reward)
                steps += 1
                done = bool(terminated or truncated)
                last_info = info

                if args.delay > 0:
                    time.sleep(args.delay)

            if last_info is None:
                continue

            print(
                f"Episode {episode_index + 1}: "
                f"reward={total_reward:.2f}, "
                f"success={last_info['mission_success']}, "
                f"detected={last_info['game_over']}, "
                f"covered={last_info['total_covered_cells']}/{last_info['coverable_cells']}, "
                f"steps_used={steps}"
            )

            time.sleep(1.0)
    finally:
        env.close()


if __name__ == "__main__":
    main()
