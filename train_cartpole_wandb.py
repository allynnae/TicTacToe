import argparse
import os
from typing import Callable

import gymnasium as gym
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from wandb.integration.sb3 import WandbCallback


def _make_env_factory(env_id: str, seed: int) -> Callable[[], gym.Env]:
    def _init() -> gym.Env:
        env = gym.make(env_id)
        env.reset(seed=seed)
        env.action_space.seed(seed)
        return env

    return _init


def train(args: argparse.Namespace) -> None:
    os.makedirs("runs", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    wandb_mode = "online" if os.getenv("WANDB_API_KEY") else "offline"
    run = wandb.init(
        project=args.project,
        entity=args.entity,
        group=args.group or None,
        notes=args.notes or None,
        config={
            "algo": "PPO",
            "env_id": args.env_id,
            "total_timesteps": args.total_timesteps,
            "n_envs": args.n_envs,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
        },
        sync_tensorboard=True,
        mode=wandb_mode,
    )

    env_fns = [_make_env_factory(args.env_id, args.seed + idx) for idx in range(args.n_envs)]
    vec_cls = SubprocVecEnv if args.n_envs > 1 else DummyVecEnv
    vec_env = vec_cls(env_fns)
    vec_env = VecMonitor(vec_env)

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=args.learning_rate,
        verbose=1,
        tensorboard_log=os.path.join("runs", run.id),
        seed=args.seed,
    )

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=WandbCallback(
            gradient_save_freq=0,
            model_save_path=None,  # Avoid symlink issues on Windows; save manually below.
            log="all",
            verbose=2,
        ),
    )

    model.save(os.path.join("models", f"{run.id}.zip"))

    vec_env.close()
    run.finish()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train PPO on a simple game (CartPole-v1) with Weights & Biases logging."
    )
    parser.add_argument("--project", type=str, default="cs5880-rl", help="W&B project name.")
    parser.add_argument("--entity", type=str, default=None, help="Optional W&B entity/org.")
    parser.add_argument("--group", type=str, default=None, help="Optional W&B run group.")
    parser.add_argument("--notes", type=str, default=None, help="Optional notes shown in W&B.")
    parser.add_argument("--env-id", type=str, default="CartPole-v1", help="Gymnasium env id.")
    parser.add_argument("--total-timesteps", type=int, default=500_000, help="Timesteps to train.")
    parser.add_argument("--n-envs", type=int, default=4, help="Parallel environments.")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Optimizer learning rate.")
    parser.add_argument("--seed", type=int, default=0, help="Base seed for reproducibility.")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
