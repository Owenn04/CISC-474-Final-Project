from __future__ import annotations

"""Analyze saved Coverage Gridworld runs and generate report-friendly outputs.

The script discovers saved models, maps them to TensorBoard logs, evaluates each
run deterministically, and exports CSV summaries plus a compact set of plots for
comparison across observation and reward variants.
"""

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path

import coverage_gridworld  # noqa: F401  # Required for Gymnasium env registration
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from coverage_gridworld import custom as custom_runtime
from stable_baselines3 import DQN, PPO
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


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

OBSERVATION_ORDER = list(custom_runtime.OBSERVATION_MODES)
REWARD_ORDER = list(custom_runtime.REWARD_MODES)
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


@dataclass
class RunArtifact:
    """Saved model plus the metadata needed for evaluation and plotting."""

    algorithm: str
    env_id: str
    observation_mode: str
    reward_mode: str
    timesteps: int
    model_path: Path
    model_mtime: float
    log_dir: Path | None = None
    log_match_method: str | None = None

    @property
    def run_id(self) -> str:
        return f"{self.algorithm}_{self.observation_mode}_{self.reward_mode}"

    @property
    def label(self) -> str:
        return f"{self.algorithm.upper()} | {self.observation_mode} | {self.reward_mode}"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for experiment analysis."""

    parser = argparse.ArgumentParser(
        description="Generate evaluation summaries and training plots for saved Coverage Gridworld runs."
    )
    parser.add_argument("--models-dir", default="artifacts/models", help="Directory containing saved .zip models.")
    parser.add_argument("--logs-dir", default="artifacts/logs", help="Directory containing TensorBoard event logs.")
    parser.add_argument("--output-dir", default="artifacts/analysis", help="Where plots and CSV summaries are written.")
    parser.add_argument("--eval-episodes", type=int, default=30, help="Number of evaluation episodes per model.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed for evaluation.")
    parser.add_argument(
        "--env-id",
        default=None,
        help="Override environment id during evaluation. By default this is read from the model filename.",
    )
    return parser.parse_args()


def build_predefined_map_list(map_ids: list[str]) -> list[list[list[int]]]:
    """Resolve registered map ids into deep-copied predefined maps for evaluation."""

    maps: list[list[list[int]]] = []
    for map_id in map_ids:
        spec = gym.spec(map_id)
        predefined_map = spec.kwargs.get("predefined_map")
        if predefined_map is None:
            raise ValueError(f"Map '{map_id}' does not expose a predefined_map and cannot be used in a map set.")
        maps.append([row[:] for row in predefined_map])
    return maps


def resolve_env_label(env_label: str) -> tuple[str, list[list[list[int]]] | None]:
    """Map a saved filename label back to an evaluable env id and optional map list."""

    base_label = env_label
    if "_rand" in env_label:
        prefix, suffix = env_label.rsplit("_rand", 1)
        if prefix in MAP_SETS and suffix.isdigit():
            base_label = prefix

    if base_label in MAP_SETS:
        return "standard", build_predefined_map_list(MAP_SETS[base_label])

    return env_label, None


def make_env(
    env_id: str,
    observation_mode: str,
    reward_mode: str,
    seed: int,
    predefined_map_list: list[list[list[int]]] | None = None,
):
    """Create an evaluation environment configured for the selected runtime modes."""

    env = gym.make(
        env_id,
        render_mode=None,
        predefined_map_list=predefined_map_list,
        activate_game_status=False,
        observation_mode=observation_mode,
        reward_mode=reward_mode,
    )
    custom_runtime.configure_runtime(env.unwrapped, observation_mode, reward_mode)
    env.unwrapped.observation_space = custom_runtime.observation_space(env.unwrapped)
    env.reset(seed=seed)
    return env


def discover_model_runs(models_dir: Path) -> list[RunArtifact]:
    """Find saved models that follow the standard training filename scheme."""

    runs: list[RunArtifact] = []
    for model_path in sorted(models_dir.glob("*.zip")):
        match = MODEL_PATTERN.match(model_path.name)
        if not match:
            continue

        runs.append(
            RunArtifact(
                algorithm=match.group("algorithm"),
                env_id=match.group("env_id"),
                observation_mode=match.group("observation_mode"),
                reward_mode=match.group("reward_mode"),
                timesteps=int(match.group("timesteps")),
                model_path=model_path,
                model_mtime=model_path.stat().st_mtime,
            )
        )

    if not runs:
        raise FileNotFoundError(f"No model files matching the expected naming scheme were found in {models_dir}")

    return runs


def event_file_mtime(log_dir: Path) -> float:
    """Return the newest TensorBoard event-file modification time for a log dir."""

    event_files = sorted(log_dir.glob("events.out.tfevents.*"))
    if not event_files:
        return -math.inf
    return max(path.stat().st_mtime for path in event_files)


def is_legacy_log_dir(path: Path) -> bool:
    """Return whether ``path`` is an old auto-numbered SB3 log directory."""

    return bool(re.fullmatch(r"(PPO|DQN)_\d+", path.name))


def assign_exact_log_dirs(runs: list[RunArtifact], logs_dir: Path) -> tuple[list[RunArtifact], list[Path]]:
    """Assign logs whose directory name exactly matches the saved model stem."""

    unassigned_runs: list[RunArtifact] = []
    used_logs: set[Path] = set()

    for run in runs:
        exact_dir = logs_dir / run.model_path.stem
        if exact_dir.is_dir():
            run.log_dir = exact_dir
            run.log_match_method = "exact_name"
            used_logs.add(exact_dir)
        else:
            unassigned_runs.append(run)

    remaining_logs = [
        path for path in logs_dir.iterdir()
        if path.is_dir() and path not in used_logs
    ]
    return unassigned_runs, remaining_logs


def assign_validated_legacy_logs(runs: list[RunArtifact], log_dirs: list[Path], max_time_delta_seconds: float = 180.0) -> None:
    """Assign old numbered log dirs only when the nearest timestamp match is unique.

    A legacy match is accepted only when:
    - algorithm matches
    - the run and log are each other's nearest candidate by event-file mtime
    - the absolute time delta is within the allowed window
    """

    for algorithm in sorted({run.algorithm for run in runs}):
        algorithm_runs = [run for run in runs if run.algorithm == algorithm and run.log_dir is None]
        algorithm_logs = [
            path for path in log_dirs
            if is_legacy_log_dir(path) and path.name.startswith(algorithm.upper())
        ]
        if not algorithm_runs or not algorithm_logs:
            continue

        log_times = {path: event_file_mtime(path) for path in algorithm_logs}
        nearest_log_for_run: dict[int, tuple[Path, float] | None] = {}
        for run in algorithm_runs:
            candidates = [
                (path, abs(log_times[path] - run.model_mtime))
                for path in algorithm_logs
                if math.isfinite(log_times[path])
            ]
            nearest_log_for_run[id(run)] = min(candidates, key=lambda item: item[1]) if candidates else None

        nearest_run_for_log: dict[Path, tuple[int, float] | None] = {}
        for path in algorithm_logs:
            candidates = [
                (id(run), abs(log_times[path] - run.model_mtime))
                for run in algorithm_runs
            ]
            nearest_run_for_log[path] = min(candidates, key=lambda item: item[1]) if candidates else None

        for run in algorithm_runs:
            log_candidate = nearest_log_for_run.get(id(run))
            if log_candidate is None:
                continue
            log_dir, delta = log_candidate
            reverse_candidate = nearest_run_for_log.get(log_dir)
            if reverse_candidate is None:
                continue
            reverse_run_id, reverse_delta = reverse_candidate
            if reverse_run_id == id(run) and delta <= max_time_delta_seconds and reverse_delta <= max_time_delta_seconds:
                run.log_dir = log_dir
                run.log_match_method = "validated_legacy_time"


def map_logs_to_runs(runs: list[RunArtifact], logs_dir: Path) -> None:
    """Match logs to runs using exact names first and validated legacy timestamps second."""

    unassigned_runs, remaining_logs = assign_exact_log_dirs(runs, logs_dir)
    assign_validated_legacy_logs(unassigned_runs, remaining_logs)


def select_scalar_tag(tags: list[str], candidates: list[str]) -> str | None:
    """Pick the first available TensorBoard scalar tag from a preferred list."""

    for candidate in candidates:
        if candidate in tags:
            return candidate
    return None


def extract_scalar_series(log_dir: Path, metric: str, candidates: list[str]) -> pd.DataFrame:
    """Extract a single scalar series from a TensorBoard log directory."""

    event_files = sorted(log_dir.glob("events.out.tfevents.*"))
    if not event_files:
        return pd.DataFrame(columns=["step", "value", "metric"])

    accumulator = EventAccumulator(str(event_files[-1]))
    accumulator.Reload()
    tag = select_scalar_tag(accumulator.Tags().get("scalars", []), candidates)
    if tag is None:
        return pd.DataFrame(columns=["step", "value", "metric"])

    scalars = accumulator.Scalars(tag)
    return pd.DataFrame(
        {
            "step": [entry.step for entry in scalars],
            "value": [entry.value for entry in scalars],
            "metric": metric,
        }
    )


def collect_training_curves(run: RunArtifact) -> pd.DataFrame:
    """Collect the training reward and episode-length curves for one run."""

    if run.log_dir is None:
        return pd.DataFrame(columns=["run_id", "algorithm", "observation_mode", "reward_mode", "metric", "step", "value"])

    reward_curve = extract_scalar_series(run.log_dir, "training_reward", ["rollout/ep_rew_mean"])
    episode_length_curve = extract_scalar_series(run.log_dir, "episode_length", ["rollout/ep_len_mean"])

    curve = pd.concat([reward_curve, episode_length_curve], ignore_index=True)
    if curve.empty:
        return pd.DataFrame(columns=["run_id", "algorithm", "observation_mode", "reward_mode", "metric", "step", "value"])

    curve["run_id"] = run.run_id
    curve["algorithm"] = run.algorithm
    curve["observation_mode"] = run.observation_mode
    curve["reward_mode"] = run.reward_mode
    curve["log_match_method"] = run.log_match_method or "unmatched"
    return curve


def build_log_assignment_df(runs: list[RunArtifact]) -> pd.DataFrame:
    """Summarize which log directory, if any, was matched to each model."""

    return pd.DataFrame(
        [
            {
                "model_name": run.model_path.name,
                "algorithm": run.algorithm,
                "env_id": run.env_id,
                "observation_mode": run.observation_mode,
                "reward_mode": run.reward_mode,
                "timesteps": run.timesteps,
                "log_dir": run.log_dir.name if run.log_dir is not None else "",
                "log_match_method": run.log_match_method or "unmatched",
            }
            for run in runs
        ]
    )


def evaluate_run(run: RunArtifact, eval_episodes: int, seed: int, env_id_override: str | None) -> pd.DataFrame:
    """Run deterministic evaluation episodes for one saved model."""

    env_label = env_id_override or run.env_id
    env_id, predefined_map_list = resolve_env_label(env_label)
    env = make_env(env_id, run.observation_mode, run.reward_mode, seed, predefined_map_list)
    model = MODEL_CLASSES[run.algorithm].load(run.model_path)

    episodes: list[dict[str, float | int | bool | str]] = []

    try:
        for episode_index in range(eval_episodes):
            obs, info = env.reset(seed=seed + episode_index)
            done = False
            total_reward = 0.0
            steps = 0
            last_info = info

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                info = custom_runtime.enrich_info(info)
                total_reward += float(reward)
                steps += 1
                done = bool(terminated or truncated)
                last_info = info

            cells_covered = int(last_info["total_covered_cells"])
            coverable_cells = max(int(last_info["coverable_cells"]), 1)
            cover_ratio = cells_covered / coverable_cells
            mission_success = bool(last_info["mission_success"])
            detected = bool(last_info["game_over"])
            steps_remaining = int(last_info["steps_remaining"])
            timeout = bool((not mission_success) and (not detected) and steps_remaining <= 0)
            coverage_per_step = cells_covered / max(steps, 1)

            episodes.append(
                {
                    "run_id": run.run_id,
                    "algorithm": run.algorithm,
                    "observation_mode": run.observation_mode,
                    "reward_mode": run.reward_mode,
                    "episode": episode_index,
                    "episode_reward": total_reward,
                    "mission_success": mission_success,
                    "detected": detected,
                    "timeout": timeout,
                    "cells_covered": cells_covered,
                    "coverable_cells": coverable_cells,
                    "cover_ratio": cover_ratio,
                    "episode_length": steps,
                    "steps_remaining": steps_remaining,
                    "coverage_per_step": coverage_per_step,
                }
            )
    finally:
        env.close()

    return pd.DataFrame(episodes)


def summarize_evaluations(episode_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-episode evaluation data into a report-friendly run summary."""

    group_keys = ["run_id", "algorithm", "observation_mode", "reward_mode"]
    grouped = (
        episode_df.groupby(group_keys, as_index=False)
        .agg(
            mean_reward=("episode_reward", "mean"),
            reward_std=("episode_reward", "std"),
            success_rate=("mission_success", "mean"),
            detection_rate=("detected", "mean"),
            timeout_rate=("timeout", "mean"),
            mean_cover_ratio=("cover_ratio", "mean"),
            cover_ratio_std=("cover_ratio", "std"),
            mean_cells_covered=("cells_covered", "mean"),
            mean_episode_length=("episode_length", "mean"),
            episode_length_std=("episode_length", "std"),
            mean_coverage_per_step=("coverage_per_step", "mean"),
            mean_steps_remaining=("steps_remaining", "mean"),
        )
    )

    success_only = (
        episode_df[episode_df["mission_success"]]
        .groupby(group_keys, as_index=False)
        .agg(mean_steps_remaining_on_success=("steps_remaining", "mean"))
    )
    failure_only = (
        episode_df[~episode_df["mission_success"]]
        .groupby(group_keys, as_index=False)
        .agg(mean_coverage_before_failure=("cover_ratio", "mean"))
    )

    grouped = grouped.merge(success_only, on=group_keys, how="left")
    grouped = grouped.merge(failure_only, on=group_keys, how="left")
    grouped = grouped.fillna(
        {
            "reward_std": 0.0,
            "cover_ratio_std": 0.0,
            "episode_length_std": 0.0,
        }
    )

    grouped["survival_rate"] = 1.0 - grouped["detection_rate"]

    score_weights = {
        "success_rate": 0.30,
        "mean_cover_ratio": 0.20,
        "mean_coverage_per_step": 0.15,
        "survival_rate": 0.15,
        "mean_coverage_before_failure": 0.10,
        "mean_steps_remaining_on_success": 0.10,
    }

    active_metrics: list[str] = []
    for column in score_weights:
        valid = grouped[column].dropna()
        if valid.empty or math.isclose(valid.min(), valid.max()):
            grouped[f"{column}_norm"] = np.nan
            continue

        grouped[f"{column}_norm"] = (grouped[column] - valid.min()) / (valid.max() - valid.min())
        grouped[f"{column}_norm"] = grouped[f"{column}_norm"].fillna(0.0)
        active_metrics.append(column)

    if not active_metrics:
        grouped["selection_score"] = 0.0
    else:
        active_weight_total = sum(score_weights[column] for column in active_metrics)
        grouped["selection_score"] = 0.0
        for column in active_metrics:
            grouped["selection_score"] += (
                score_weights[column] / active_weight_total
            ) * grouped[f"{column}_norm"]

    return grouped.sort_values("selection_score", ascending=False).reset_index(drop=True)


def plot_training_curves(training_df: pd.DataFrame, output_dir: Path) -> None:
    """Plot smoothed training reward and episode-length curves by algorithm."""

    if training_df.empty:
        return

    for metric, ylabel, filename in [
        ("training_reward", "Episode Reward Mean", "training_reward_over_time.png"),
        ("episode_length", "Episode Length Mean", "training_episode_length_over_time.png"),
    ]:
        metric_df = training_df[training_df["metric"] == metric].copy()
        if metric_df.empty:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True)
        for axis, algorithm in zip(axes, ["ppo", "dqn"]):
            algorithm_df = metric_df[metric_df["algorithm"] == algorithm]
            for run_id, run_df in algorithm_df.groupby("run_id"):
                run_df = run_df.sort_values("step").copy()
                run_df["smoothed"] = run_df["value"].rolling(window=5, min_periods=1).mean()
                label = f"{run_df['observation_mode'].iat[0]} | {run_df['reward_mode'].iat[0]}"
                axis.plot(run_df["step"], run_df["smoothed"], label=label, linewidth=2)

            axis.set_title(f"{algorithm.upper()} {metric.replace('_', ' ').title()}")
            axis.set_xlabel("Training Step")
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.3)
            axis.legend(fontsize=8)

        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=200, bbox_inches="tight")
        plt.close(fig)


def annotate_heatmap(axis, matrix: np.ndarray) -> None:
    """Write numeric labels directly into a heatmap matrix."""

    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            label = "NA" if np.isnan(value) else f"{value:.2f}"
            axis.text(column_index, row_index, label, ha="center", va="center", color="black", fontsize=10)


def plot_metric_heatmaps(summary_df: pd.DataFrame, output_dir: Path) -> None:
    """Render metric heatmaps across observation and reward combinations."""

    metrics = [
        ("success_rate", "Evaluation Success Rate", "heatmap_success_rate.png"),
        ("mean_cover_ratio", "Average Coverage Ratio", "heatmap_cover_ratio.png"),
        ("mean_coverage_per_step", "Coverage Per Step", "heatmap_coverage_per_step.png"),
        ("survival_rate", "Survival Rate", "heatmap_survival_rate.png"),
        ("mean_coverage_before_failure", "Coverage Before Failure", "heatmap_coverage_before_failure.png"),
        ("mean_steps_remaining_on_success", "Steps Remaining On Success", "heatmap_steps_remaining_on_success.png"),
        ("selection_score", "Composite Selection Score", "heatmap_selection_score.png"),
    ]

    for metric, title, filename in metrics:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for axis, algorithm in zip(axes, ["ppo", "dqn"]):
            subset = summary_df[summary_df["algorithm"] == algorithm]
            pivot = (
                subset.pivot(index="observation_mode", columns="reward_mode", values=metric)
                .reindex(index=OBSERVATION_ORDER, columns=REWARD_ORDER)
            )
            matrix = pivot.to_numpy(dtype=float)
            image = axis.imshow(matrix, cmap="YlGnBu", aspect="auto")
            axis.set_title(f"{algorithm.upper()} {title}")
            axis.set_xticks(range(len(REWARD_ORDER)))
            axis.set_xticklabels(REWARD_ORDER)
            axis.set_yticks(range(len(OBSERVATION_ORDER)))
            axis.set_yticklabels(OBSERVATION_ORDER)
            axis.set_xlabel("Reward Mode")
            axis.set_ylabel("Observation Mode")
            annotate_heatmap(axis, matrix)
            fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=200, bbox_inches="tight")
        plt.close(fig)


def plot_model_ranking(summary_df: pd.DataFrame, output_dir: Path) -> None:
    """Plot the composite selection score ranking for all discovered runs."""

    ranking = summary_df.sort_values("selection_score", ascending=True)

    fig, axis = plt.subplots(figsize=(12, 7))
    colors = ranking["algorithm"].map({"ppo": "#1f77b4", "dqn": "#ff7f0e"})
    axis.barh(ranking["run_id"], ranking["selection_score"], color=colors)
    axis.set_title("Model Ranking by Composite Selection Score")
    axis.set_xlabel("Selection Score")
    axis.set_ylabel("Run")
    axis.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "model_ranking.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_tradeoff_scatter(summary_df: pd.DataFrame, output_dir: Path) -> None:
    """Plot the tradeoff between average coverage and success rate."""

    fig, axis = plt.subplots(figsize=(10, 7))

    markers = {"ppo": "o", "dqn": "s"}
    colors = {
        "sparse": "#d62728",
        "coverage": "#2ca02c",
        "safety": "#9467bd",
        "baseline_coverage": "#8c564b",
        "baseline_reward_v1": "#1f77b4",
        "baseline_reward_v2": "#ff7f0e",
        "baseline_reward_v3": "#17becf",
        "baseline_reward_v4": "#bcbd22",
    }

    for _, row in summary_df.iterrows():
        axis.scatter(
            row["mean_cover_ratio"],
            row["success_rate"],
            s=max(row["mean_coverage_per_step"] * 350, 20),
            marker=markers[row["algorithm"]],
            color=colors.get(row["reward_mode"], "#7f7f7f"),
            alpha=0.8,
        )
        axis.text(
            row["mean_cover_ratio"] + 0.003,
            row["success_rate"] + 0.003,
            f"{row['algorithm'].upper()} | {row['observation_mode']} | {row['reward_mode']}",
            fontsize=8,
        )

    axis.set_title("Coverage vs Success Rate")
    axis.set_xlabel("Average Coverage Ratio")
    axis.set_ylabel("Evaluation Success Rate")
    axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "coverage_success_tradeoff.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_episode_distributions(episode_df: pd.DataFrame, output_dir: Path) -> None:
    """Plot boxplots of per-episode coverage ratios for each run."""

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    for axis, algorithm in zip(axes, ["ppo", "dqn"]):
        subset = episode_df[episode_df["algorithm"] == algorithm].copy()
        if subset.empty:
            continue

        positions = []
        labels = []
        data = []

        ordered_runs = (
            subset[["run_id", "observation_mode", "reward_mode"]]
            .drop_duplicates()
            .sort_values(["observation_mode", "reward_mode"])
        )
        for position, (_, row) in enumerate(ordered_runs.iterrows(), start=1):
            run_values = subset[subset["run_id"] == row["run_id"]]["cover_ratio"].to_numpy()
            positions.append(position)
            labels.append(f"{row['observation_mode']}\n{row['reward_mode']}")
            data.append(run_values)

        axis.boxplot(data, positions=positions, widths=0.6)
        axis.set_title(f"{algorithm.upper()} Coverage Ratio Distribution")
        axis.set_xlabel("Observation / Reward")
        axis.set_ylabel("Coverage Ratio")
        axis.set_xticks(positions)
        axis.set_xticklabels(labels, rotation=20)
        axis.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "coverage_ratio_distributions.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Generate CSV summaries and plots for all discovered saved runs."""

    args = parse_args()

    models_dir = Path(args.models_dir)
    logs_dir = Path(args.logs_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = discover_model_runs(models_dir)
    map_logs_to_runs(runs, logs_dir)
    log_assignment_df = build_log_assignment_df(runs)

    training_curves = pd.concat([collect_training_curves(run) for run in runs], ignore_index=True)
    episode_df = pd.concat(
        [evaluate_run(run, args.eval_episodes, args.seed, args.env_id) for run in runs],
        ignore_index=True,
    )
    summary_df = summarize_evaluations(episode_df)

    summary_df.to_csv(output_dir / "evaluation_summary.csv", index=False)
    episode_df.to_csv(output_dir / "evaluation_episodes.csv", index=False)
    training_curves.to_csv(output_dir / "training_curves.csv", index=False)
    log_assignment_df.to_csv(output_dir / "log_assignments.csv", index=False)

    plot_training_curves(training_curves, output_dir)
    plot_metric_heatmaps(summary_df, output_dir)
    plot_model_ranking(summary_df, output_dir)
    plot_tradeoff_scatter(summary_df, output_dir)
    plot_episode_distributions(episode_df, output_dir)

    print(f"Wrote analysis outputs to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
