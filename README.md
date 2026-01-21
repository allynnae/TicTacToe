## TicTacToe

### Overview
- This is a TicTacToe game where a human player competes against an AI bot. The player places their mark on a 3×3 grid, and the AI responds with its own strategic move. The objective is to be the first to align three marks in a row - horizontally, vertically, or diagonally - while preventing the opponent from doing the same.
- This project is set up to use gpt-4o-mini.

### Installation
Set up the virtual environment and install requirements:
```
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```
Run the program:
```
./run_tictactoe.sh
```
You will then be prompted to add your OPENAI_API_KEY and WANDB_API_KEY. 
