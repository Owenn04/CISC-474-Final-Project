from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import coverage_gridworld 
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor


SUPPORTED_ALGORITHMS = {
    "ppo": PPO,
    "dqn": DQN,
}


def parse_args() -> argparse.Namespace:
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
        choices=["full_grid", "compact"],
        default="full_grid",
        help="Observation mode exposed by the environment.",
    )
    parser.add_argument(
        "--reward-mode",
        choices=["sparse", "coverage", "safety"],
        default="coverage",
        help="Reward mode exposed by the environment.",
    )
    return parser.parse_args()


def make_env(
    env_id: str,
    *,
    seed: int,
    observation_mode: str,
    reward_mode: str,
    render: bool = False,
) -> Monitor:
    render_mode = "human" if render else None
    env = gym.make(
        env_id,
        render_mode=render_mode,
        predefined_map_list=None,
        activate_game_status=False,
        observation_mode=observation_mode,
        reward_mode=reward_mode,
    )
    env.reset(seed=seed)
    return Monitor(env)


def build_model(algorithm: str, env: Monitor, log_dir: Path, seed: int):
    model_class = SUPPORTED_ALGORITHMS[algorithm]

    common_kwargs = {
        "env": env,
        "verbose": 1,
        "seed": seed,
        "tensorboard_log": str(log_dir),
    }

    if algorithm == "ppo":
        return model_class("MlpPolicy", **common_kwargs)

    if algorithm == "dqn":
        return model_class("MlpPolicy", learning_starts=1_000, buffer_size=50_000, **common_kwargs)

    raise ValueError(f"Unsupported algorithm: {algorithm}")


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    model_dir = output_dir / "models"
    log_dir = output_dir / "logs"
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    train_env = make_env(
        args.env_id,
        seed=args.seed,
        observation_mode=args.observation_mode,
        reward_mode=args.reward_mode,
    )
    model = build_model(args.algorithm, train_env, log_dir, args.seed)
    model.learn(total_timesteps=args.timesteps, progress_bar=True)

    model_path = model_dir / (
        f"{args.algorithm}_{args.env_id}_{args.observation_mode}_{args.reward_mode}_{args.timesteps}.zip"
    )
    model.save(model_path)

    eval_env = make_env(
        args.env_id,
        seed=args.seed,
        observation_mode=args.observation_mode,
        reward_mode=args.reward_mode,
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
