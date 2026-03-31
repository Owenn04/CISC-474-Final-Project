from __future__ import annotations

"""Train Stable Baselines 3 agents on Coverage Gridworld.

This script is the main training entry point used throughout the project. It
keeps the immutable ``env.py`` untouched and configures custom observation and
reward behavior through ``coverage_gridworld.custom`` at runtime.
"""

import argparse
import copy
from pathlib import Path

import gymnasium as gym
import coverage_gridworld  # noqa: F401  # Required for Gymnasium env registration
from coverage_gridworld import custom as custom_runtime
import torch as th
from torch import nn
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


SUPPORTED_ALGORITHMS = {
    "ppo": PPO,
    "dqn": DQN,
}

MAP_SETS = {
    "all_standard_maps": ["just_go", "safe", "maze", "chokepoint", "sneaky_enemies"],
    "all_standard_plus_custom": ["just_go", "safe", "maze", "custom_challenge", "chokepoint", "sneaky_enemies"],
    "coverage_curriculum": ["just_go", "safe", "maze"],
    "enemy_mix": [
        "maze",
        "custom_challenge",
        "timing_corridor",
        "staggered_escape",
        "chokepoint",
        "sneaky_enemies",
    ],
    "enemy_mix_large": [
        "maze",
        "custom_challenge",
        "timing_corridor",
        "staggered_escape",
        "patrol_weave",
        "enemy_spine",
        "sidepass_patrol",
        "chokepoint",
        "sneaky_enemies",
    ],
    "frontier_mix_large": [
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
    ],
    "generalization_train": ["safe", "maze", "chokepoint"],
    "generalization_test": ["sneaky_enemies"],
}


class SmallGridCNN(BaseFeaturesExtractor):
    """Small CNN used when the observation is emitted as a grid tensor."""

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with th.no_grad():
            sample = th.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations.float()))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for training and post-training evaluation."""

    parser = argparse.ArgumentParser(
        description="Train a Stable Baselines 3 agent on Coverage Gridworld."
    )
    parser.add_argument(
        "--algorithm",
        choices=sorted(SUPPORTED_ALGORITHMS.keys()),
        default="ppo",
        help="RL algorithm to train.",
    )
    parser.add_argument(
        "--env-id",
        default="standard",
        help="Gymnasium environment ID registered by coverage_gridworld.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=50_000,
        help="Number of training timesteps.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes after training.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the environment during evaluation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for training and evaluation.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts",
        help="Directory where models and logs will be saved.",
    )
    parser.add_argument(
        "--observation-mode",
        choices=custom_runtime.OBSERVATION_MODES,
        default="full_grid",
        help="Observation mode exposed by the environment.",
    )
    parser.add_argument(
        "--reward-mode",
        choices=custom_runtime.REWARD_MODES,
        default="coverage",
        help="Reward mode exposed by the environment.",
    )
    parser.add_argument(
        "--map-set",
        choices=sorted(MAP_SETS.keys()),
        default=None,
        help="Optional predefined training map set. When provided, training uses env-id='standard' with a rotating predefined_map_list.",
    )
    parser.add_argument(
        "--eval-env-id",
        default=None,
        help="Optional environment id for evaluation. Defaults to the training env id, or 'standard' if --map-set is used.",
    )
    parser.add_argument(
        "--random-standard-prob",
        type=float,
        default=0.0,
        help="Not supported with the immutable env.py setup. Leave at 0.0.",
    )
    return parser.parse_args()


def build_predefined_map_list(map_ids: list[str]) -> list[list[list[int]]]:
    """Resolve registered map ids into deep-copied predefined maps for rotation."""

    maps: list[list[list[int]]] = []
    for map_id in map_ids:
        spec = gym.spec(map_id)
        predefined_map = spec.kwargs.get("predefined_map")
        if predefined_map is None:
            raise ValueError(f"Map '{map_id}' does not expose a predefined_map and cannot be used in a map set.")
        maps.append(copy.deepcopy(predefined_map))
    return maps


def make_env(
    env_id: str,
    *,
    seed: int,
    observation_mode: str,
    reward_mode: str,
    predefined_map_list: list[list[list[int]]] | None = None,
    random_standard_prob: float = 0.0,
    render: bool = False,
) -> Monitor:
    """Create a monitored environment configured for the selected runtime modes."""

    if random_standard_prob > 0.0:
        raise ValueError("random_standard_prob is not supported without modifying env.py or using wrappers.")
    render_mode = "human" if render else None
    env = gym.make(
        env_id,
        render_mode=render_mode,
        predefined_map_list=predefined_map_list,
        activate_game_status=False,
        observation_mode=observation_mode,
        reward_mode=reward_mode,
    )
    custom_runtime.configure_runtime(env.unwrapped, observation_mode, reward_mode)
    env.unwrapped.observation_space = custom_runtime.observation_space(env.unwrapped)
    env.reset(seed=seed)
    return Monitor(env)


def build_model(algorithm: str, env: Monitor, log_dir: Path, seed: int):
    """Instantiate the requested SB3 model with sensible defaults for the task."""

    model_class = SUPPORTED_ALGORITHMS[algorithm]
    observation_shape = getattr(env.observation_space, "shape", ())
    policy = "CnnPolicy" if len(observation_shape) == 3 else "MlpPolicy"

    common_kwargs = {
        "env": env,
        "verbose": 1,
        "seed": seed,
        "tensorboard_log": str(log_dir),
    }
    if policy == "CnnPolicy":
        common_kwargs["policy_kwargs"] = {
            "features_extractor_class": SmallGridCNN,
            "features_extractor_kwargs": {"features_dim": 128},
        }

    if algorithm == "ppo":
        return model_class(
            policy,
            ent_coef=0.01,
            policy_kwargs={"net_arch": [256, 256]},
            **common_kwargs,
        )

    if algorithm == "dqn":
        return model_class(policy, learning_starts=1_000, buffer_size=50_000, **common_kwargs)

    raise ValueError(f"Unsupported algorithm: {algorithm}")


def latest_matching_log_dir(log_root: Path, run_name: str) -> Path | None:
    """Return the newest SB3 TensorBoard directory created for ``run_name``."""

    matches = [
        path
        for path in log_root.glob(f"{run_name}*")
        if path.is_dir()
    ]
    if not matches:
        return None
    return max(matches, key=lambda path: path.stat().st_mtime)


def finalize_log_dir(log_root: Path, run_name: str) -> Path | None:
    """Rename the SB3-created TensorBoard directory to an exact descriptive name."""

    created_log_dir = latest_matching_log_dir(log_root, run_name)
    if created_log_dir is None:
        return None

    target_dir = log_root / run_name
    if created_log_dir == target_dir:
        return target_dir

    if target_dir.exists():
        return target_dir

    created_log_dir.rename(target_dir)
    return target_dir


def main() -> None:
    """Train a model, save it, and run a deterministic evaluation pass."""

    args = parse_args()

    output_dir = Path(args.output_dir)
    model_dir = output_dir / "models"
    log_dir = output_dir / "logs"
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    predefined_map_list = None
    train_env_id = args.env_id
    if args.map_set is not None:
        predefined_map_list = build_predefined_map_list(MAP_SETS[args.map_set])
        train_env_id = "standard"

    env_label = args.map_set or args.env_id
    if args.map_set is not None and args.random_standard_prob > 0.0:
        env_label = f"{env_label}_rand{int(args.random_standard_prob * 100):02d}"
    run_name = f"{args.algorithm}_{env_label}_{args.observation_mode}_{args.reward_mode}_{args.timesteps}"

    train_env = make_env(
        train_env_id,
        seed=args.seed,
        observation_mode=args.observation_mode,
        reward_mode=args.reward_mode,
        predefined_map_list=predefined_map_list,
        random_standard_prob=args.random_standard_prob,
    )
    model = build_model(args.algorithm, train_env, log_dir, args.seed)
    model.learn(total_timesteps=args.timesteps, progress_bar=True, tb_log_name=run_name)

    model_path = model_dir / f"{run_name}.zip"
    model.save(model_path)
    finalize_log_dir(log_dir, run_name)

    eval_env_id = args.eval_env_id or train_env_id
    eval_predefined_map_list = predefined_map_list if eval_env_id == "standard" else None
    eval_env = make_env(
        eval_env_id,
        seed=args.seed,
        observation_mode=args.observation_mode,
        reward_mode=args.reward_mode,
        predefined_map_list=eval_predefined_map_list,
        random_standard_prob=0.0,
        render=args.render,
    )
    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
    )

    print(f"Saved model to: {model_path}")
    print(f"Evaluation mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
