import argparse
import os
from typing import Any, Callable, Dict

import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from wandb.integration.sb3 import WandbCallback

from tictactoe_env import TicTacToeEnv


def build_vec_env(args: argparse.Namespace) -> VecMonitor:
    env_kwargs: Dict[str, Any] = {
        "invalid_penalty": args.invalid_penalty,
        "draw_reward": args.draw_reward,
        "step_penalty": args.step_penalty,
        "opponent_first_prob": args.opponent_first_prob,
    }

    def _make_env(rank: int) -> Callable[[], TicTacToeEnv]:
        def _init() -> TicTacToeEnv:
            env = TicTacToeEnv(**env_kwargs)
            env.reset(seed=args.seed + rank)
            env.action_space.seed(args.seed + rank)
            return env

        return _init

    env_fns = [_make_env(idx) for idx in range(args.n_envs)]
    return VecMonitor(DummyVecEnv(env_fns))


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
            "env": "TicTacToeEnv",
            "total_timesteps": args.total_timesteps,
            "n_envs": args.n_envs,
            "learning_rate": args.learning_rate,
            "n_steps": args.n_steps,
            "batch_size": args.batch_size,
            "gamma": args.gamma,
            "gae_lambda": args.gae_lambda,
            "clip_range": args.clip_range,
            "ent_coef": args.ent_coef,
            "vf_coef": args.vf_coef,
            "seed": args.seed,
            "invalid_penalty": args.invalid_penalty,
            "draw_reward": args.draw_reward,
            "step_penalty": args.step_penalty,
            "opponent_first_prob": args.opponent_first_prob,
        },
        sync_tensorboard=True,
        mode=wandb_mode,
    )

    vec_env = build_vec_env(args)

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        verbose=1,
        tensorboard_log=os.path.join("runs", run.id),
        seed=args.seed,
    )

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=WandbCallback(
            gradient_save_freq=0,
            model_save_path=None,
            log="all",
            verbose=2,
        ),
    )

    model.save(os.path.join("models", f"{run.id}.zip"))
    vec_env.close()
    run.finish()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train PPO on TicTacToe with Weights & Biases logging."
    )
    parser.add_argument("--project", type=str, default="cs5880-tictactoe", help="W&B project name.")
    parser.add_argument("--entity", type=str, default="am893120", help="W&B entity/user.")
    parser.add_argument("--group", type=str, default="ppo-tictactoe", help="Optional W&B run group.")
    parser.add_argument("--notes", type=str, default=None, help="Optional W&B notes.")
    parser.add_argument("--total-timesteps", type=int, default=500_000, help="Timesteps to train.")
    parser.add_argument("--n-envs", type=int, default=8, help="Parallel environments.")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Optimizer learning rate.")
    parser.add_argument("--n-steps", type=int, default=256, help="Rollout steps per environment.")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size for PPO updates.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda.")
    parser.add_argument("--clip-range", type=float, default=0.2, help="PPO clip range.")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="Entropy loss coefficient.")
    parser.add_argument("--vf-coef", type=float, default=0.5, help="Value function loss coefficient.")
    parser.add_argument("--invalid-penalty", type=float, default=-1.0, help="Reward for illegal moves.")
    parser.add_argument("--draw-reward", type=float, default=0.1, help="Reward for a draw.")
    parser.add_argument("--step-penalty", type=float, default=-0.01, help="Reward per non-terminal step.")
    parser.add_argument(
        "--opponent-first-prob",
        type=float,
        default=0.5,
        help="Probability that the opponent moves first after reset.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
