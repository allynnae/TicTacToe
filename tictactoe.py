import json
import os
import re
import threading
import tkinter as tk
import urllib.error
import urllib.request
from typing import List, Optional


HUMAN = "X"
AI = "O"
EMPTY = ""
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "").strip() or "gpt-4o-mini"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

def _parse_timeout(val: str, default: float) -> float:
    try:
        v = float(val)
        if v <= 0:
            raise ValueError
        return v
    except Exception:
        return default

OPENAI_TIMEOUT = _parse_timeout(os.getenv("OPENAI_TIMEOUT", ""), 30.0)


class TicTacToeApp:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Tic Tac Toe")
        self.root.resizable(False, False)
        self.colors = {
            "bg": "#0b1021",
            "panel": "#11182a",
            "border": "#1f2937",
            "grid": "#1c2540",
            "cell": "#161f35",
            "cell_hover": "#1f2a44",
            "cell_active": "#253252",
            "text": "#e6eaf3",
            "muted": "#9aa7b8",
            "x": "#5be7c4",
            "o": "#f472b6",
            "button": "#3b82f6",
            "button_hover": "#2563eb",
        }

        self.board: List[str] = [EMPTY for _ in range(9)]
        self.buttons: List[tk.Button] = []
        self.status_var = tk.StringVar(value="Your turn (X)")
        self.game_active = True

        self._build_ui()

    def _build_ui(self) -> None:
        self.root.configure(bg=self.colors["bg"])
        self.root.option_add("*Font", ("Segoe UI", 11))

        container = tk.Frame(self.root, bg=self.colors["bg"], padx=20, pady=20)
        container.grid(row=0, column=0)

        title = tk.Label(
            container,
            text="Tic Tac Toe",
            fg=self.colors["text"],
            bg=self.colors["bg"],
            font=("Segoe UI Semibold", 18),
        )
        title.grid(row=0, column=0, sticky="w", pady=(0, 14))

        card = tk.Frame(
            container,
            bg=self.colors["panel"],
            padx=12,
            pady=12,
            bd=0,
            highlightthickness=1,
            highlightbackground=self.colors["border"],
        )
        card.grid(row=1, column=0, sticky="nsew")

        board_frame = tk.Frame(card, bg=self.colors["grid"], padx=10, pady=10)
        board_frame.grid(row=0, column=0)

        for idx in range(9):
            row, col = divmod(idx, 3)
            btn = tk.Button(
                board_frame,
                text="",
                width=4,
                height=2,
                font=("Segoe UI", 22, "bold"),
                command=lambda i=idx: self.on_cell_click(i),
            )
            btn.grid(row=row, column=col, padx=6, pady=6)
            self._style_cell_button(btn)
            self.buttons.append(btn)

        status_label = tk.Label(
            container,
            textvariable=self.status_var,
            font=("Segoe UI", 11),
            fg=self.colors["text"],
            bg=self.colors["bg"],
            padx=6,
            pady=4,
        )
        status_label.grid(row=2, column=0, sticky="w", pady=(12, 8))

        reset_btn = tk.Button(
            container,
            text="New Game",
            command=self.reset_board,
            fg=self.colors["text"],
            bg=self.colors["button"],
            activebackground=self.colors["button_hover"],
            activeforeground=self.colors["text"],
            relief="flat",
            borderwidth=0,
            padx=12,
            pady=6,
            font=("Segoe UI Semibold", 11),
            cursor="hand2",
            highlightthickness=0,
        )
        reset_btn.grid(row=3, column=0, sticky="w", pady=(0, 4))
        reset_btn.bind(
            "<Enter>", lambda _e, b=reset_btn: b.config(bg=self.colors["button_hover"])
        )
        reset_btn.bind(
            "<Leave>", lambda _e, b=reset_btn: b.config(bg=self.colors["button"])
        )

    def reset_board(self) -> None:
        self.board = [EMPTY for _ in range(9)]
        for btn in self.buttons:
            btn.config(text="", state=tk.NORMAL)
            self._style_cell_button(btn)
        self.status_var.set("Your turn (X)")
        self.game_active = True

    def on_cell_click(self, idx: int) -> None:
        if not self.game_active or self.board[idx] != EMPTY:
            return

        self._place_mark(idx, HUMAN)
        if self._check_end_state():
            return

        self.status_var.set("AI thinking...")
        self.game_active = False
        threading.Thread(target=self._ai_move_thread, daemon=True).start()

    def _ai_move_thread(self) -> None:
        move = request_openai_move(
            self.board,
            model=OPENAI_MODEL,
            api_key=OPENAI_API_KEY,
            timeout=OPENAI_TIMEOUT,
        )
        self.root.after(0, lambda: self._apply_ai_move(move))

    def _apply_ai_move(self, move: Optional[int]) -> None:
        if self._check_end_state():
            return

        chosen_move = move
        if chosen_move is None or chosen_move not in range(9) or self.board[chosen_move] != EMPTY:
            print("[AI] Using fallback minimax.")
            chosen_move = self._best_move_fallback()

        if chosen_move is not None:
            self._place_mark(chosen_move, AI)

        if not self._check_end_state():
            self.status_var.set("Your turn (X)")
            self.game_active = True

    def _place_mark(self, idx: int, mark: str) -> None:
        self.board[idx] = mark
        fg = self.colors["x"] if mark == HUMAN else self.colors["o"]
        self.buttons[idx].config(
            text=mark,
            state=tk.DISABLED,
            fg=fg,
            disabledforeground=fg,
        )

    def _check_end_state(self) -> bool:
        winner = check_winner(self.board)
        if winner:
            self.status_var.set("You win!" if winner == HUMAN else "Computer wins.")
            self._lock_board()
            return True
        if is_draw(self.board):
            self.status_var.set("Draw game.")
            self._lock_board()
            return True
        return False

    def _lock_board(self) -> None:
        self.game_active = False
        for btn in self.buttons:
            btn.config(state=tk.DISABLED)

    def _best_move_fallback(self) -> Optional[int]:
        best_score = float("-inf")
        move: Optional[int] = None
        for idx, cell in enumerate(self.board):
            if cell == EMPTY:
                self.board[idx] = AI
                score = minimax(self.board, False)
                self.board[idx] = EMPTY
                if score > best_score:
                    best_score = score
                    move = idx
        return move

    def _style_cell_button(self, btn: tk.Button) -> None:
        btn.config(
            bg=self.colors["cell"],
            fg=self.colors["text"],
            activebackground=self.colors["cell_active"],
            activeforeground=self.colors["text"],
            disabledforeground=self.colors["muted"],
            borderwidth=0,
            relief="flat",
            highlightthickness=0,
            cursor="hand2",
        )
        btn.unbind("<Enter>")
        btn.unbind("<Leave>")

        def on_enter(event: tk.Event) -> None:  # type: ignore[override]
            if str(btn["state"]) != tk.DISABLED:
                btn.config(bg=self.colors["cell_hover"])

        def on_leave(event: tk.Event) -> None:  # type: ignore[override]
            if str(btn["state"]) != tk.DISABLED:
                btn.config(bg=self.colors["cell"])

        btn.bind("<Enter>", on_enter)
        btn.bind("<Leave>", on_leave)

    def run(self) -> None:
        self.root.mainloop()


def check_winner(board: List[str]) -> Optional[str]:
    winning_lines = [
        (0, 1, 2),
        (3, 4, 5),
        (6, 7, 8),
        (0, 3, 6),
        (1, 4, 7),
        (2, 5, 8),
        (0, 4, 8),
        (2, 4, 6),
    ]
    for a, b, c in winning_lines:
        if board[a] and board[a] == board[b] == board[c]:
            return board[a]
    return None


def is_draw(board: List[str]) -> bool:
    return all(cell != EMPTY for cell in board) and check_winner(board) is None


def minimax(board: List[str], maximizing: bool) -> int:
    """
    Unbeatable search: maximizing chooses AI moves, minimizing chooses human moves.
    Returns 1 for an AI win, -1 for a human win, and 0 for a draw.
    """
    winner = check_winner(board)
    if winner == AI:
        return 1
    if winner == HUMAN:
        return -1
    if is_draw(board):
        return 0

    if maximizing:
        best_score = float("-inf")
        for idx, cell in enumerate(board):
            if cell == EMPTY:
                board[idx] = AI
                score = minimax(board, False)
                board[idx] = EMPTY
                best_score = max(best_score, score)
        return best_score

    best_score = float("inf")
    for idx, cell in enumerate(board):
        if cell == EMPTY:
            board[idx] = HUMAN
            score = minimax(board, True)
            board[idx] = EMPTY
            best_score = min(best_score, score)
    return best_score


def request_openai_move(
    board: List[str], model: str, api_key: str, timeout: float
) -> Optional[int]:
    if not api_key:
        print("[AI] OPENAI_API_KEY not set; using fallback minimax.")
        return None

    prompt = build_prompt(board)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are an unbeatable tic tac toe player playing as O."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
    }
    try:
        print(f"[AI] Querying OpenAI model '{model}'...")
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        text = (
            result.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        print("[AI] OpenAI response:")
        print(text.strip())
        return parse_move(text, board)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore") if hasattr(exc, "read") else ""
        print(f"[AI] OpenAI HTTP error ({exc.code}): {body or exc}")
        return None
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        print(f"[AI] OpenAI request failed, using fallback. Reason: {exc}")
        return None


def build_prompt(board: List[str]) -> str:
    slots = []
    for idx, cell in enumerate(board):
        display = cell if cell else "."
        slots.append(f"{idx}:{display}")
    board_state = " ".join(slots)
    return (
        "You are playing tic tac toe as O. I am X. "
        "Board is indexed 0-8 left-to-right, top-to-bottom. "
        f"Current board: {board_state}. "
        "Think briefly about the best EMPTY position, explain your reasoning in one or two sentences, "
        "then output a final line exactly as: MOVE: <index> using digits 0-8."
    )


def parse_move(response_text: str, board: List[str]) -> Optional[int]:
    move_line = re.search(r"MOVE\s*:\s*([0-8])", response_text, re.IGNORECASE)
    if move_line:
        idx = int(move_line.group(1))
        if board[idx] == EMPTY:
            return idx

    candidates = re.findall(r"\b([0-8])\b", response_text)
    for cand in candidates:
        idx = int(cand)
        if board[idx] == EMPTY:
            return idx
    return None


if __name__ == "__main__":
    TicTacToeApp().run()
