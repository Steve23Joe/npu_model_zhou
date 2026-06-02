"""Train the Stage 3B Stable-Baselines3 DQN offloading policy."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from src.env.offloading_env import OffloadingEnv


def load_training_config() -> dict[str, object]:
    """Load Stage 3B training configuration."""
    path = ROOT / "configs" / "train_config.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def main() -> None:
    config = load_training_config()
    output_dir = ROOT / str(config.get("output_dir", "results/rl"))
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "stage3b_dqn_model.zip"
    compatibility_model_path = output_dir / "stage3_dqn_model.zip"
    metadata_path = output_dir / "stage3b_dqn_training_metadata.json"
    eval_dir = output_dir / "stage3b_eval_during_training"
    eval_dir.mkdir(parents=True, exist_ok=True)

    seed = int(config["seed"])
    max_steps = 200
    env = OffloadingEnv(seed=seed, max_steps=max_steps)
    check_env(env, warn=True)
    train_env = Monitor(OffloadingEnv(seed=seed, max_steps=max_steps))
    eval_env = Monitor(OffloadingEnv(seed=seed + 1, max_steps=max_steps))
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(eval_dir),
        log_path=str(eval_dir),
        eval_freq=int(config["eval_freq"]),
        n_eval_episodes=int(config["eval_episodes"]),
        deterministic=True,
        render=False,
        verbose=0,
    )

    started = time.perf_counter()
    model = DQN(
        "MlpPolicy",
        train_env,
        learning_rate=float(config["learning_rate"]),
        buffer_size=int(config["buffer_size"]),
        learning_starts=int(config["learning_starts"]),
        batch_size=int(config["batch_size"]),
        gamma=float(config["gamma"]),
        train_freq=int(config["train_freq"]),
        target_update_interval=int(config["target_update_interval"]),
        exploration_fraction=float(config["exploration_fraction"]),
        exploration_final_eps=float(config["exploration_final_eps"]),
        policy_kwargs={"net_arch": [64, 64]},
        seed=seed,
        verbose=0,
    )
    model.learn(
        total_timesteps=int(config["total_timesteps"]),
        callback=eval_callback,
        progress_bar=False,
    )
    training_seconds = time.perf_counter() - started
    model.save(str(model_path))
    model.save(str(compatibility_model_path))

    metadata = {
        "stage": "3B",
        "model_path": str(model_path),
        "compatibility_model_path": str(compatibility_model_path),
        "best_model_dir": str(eval_dir),
        "training_seconds": training_seconds,
        "config": config,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Stage 3B DQN training complete")
    print(f"Saved model: {model_path}")
    print(f"Saved compatibility model: {compatibility_model_path}")
    print(f"Saved metadata: {metadata_path}")
    print(f"Training seconds: {training_seconds:.2f}")


if __name__ == "__main__":
    main()
