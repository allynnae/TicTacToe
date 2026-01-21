## RL + Weights & Biases

### TicTacToe (PPO vs heuristic opponent)
- Custom Gymnasium env (`TicTacToeEnv`) where the agent plays as X, gets `+1` for wins, `-1` for losses, small negative step cost, and a penalty for illegal moves; the opponent uses a simple win/block/center/corner heuristic with random starts.
- WSL/Linux shortcut: `./run_tictactoe.sh` (prompts for keys; defaults seeds 0,1,2,3, project `cs5880-tictactoe`, entity `am893120`, group `ppo-tictactoe`, 500k steps, 8 envs). Override via env vars like `SEEDS="0,1,2" WANDB_KEY="..." ./run_tictactoe.sh`.
- Quick start: `.\run_tictactoe.ps1 -OpenAIKey "<key>" -WandbKey "<key>" -Seeds "0,1,2,3"` (defaults to project `cs5880-tictactoe`, entity `am893120`, group `ppo-tictactoe`, 500k steps, 8 envs). If you omit keys, W&B falls back to offline mode and OpenAI calls are skipped.
- Direct run (if you prefer): `python train_tictactoe_wandb.py --project cs5880-tictactoe --entity am893120 --group ppo-tictactoe --total-timesteps 500000 --n-envs 8 --seed 0`. Use multiple seeds (e.g., 0-3) to produce overlapping W&B curves like the screenshots. Models save under `models/<run-id>.zip` and TensorBoard logs under `runs/<run-id>/`.
- Tunables: `--opponent-first-prob` controls how often the opponent starts; tweak `--invalid-penalty`, `--draw-reward`, or `--step-penalty` to shape behavior. Defaults fall back to W&B offline mode when `WANDB_API_KEY` is unset.
