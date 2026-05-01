from minesweeper import main
from pathlib import Path

ROWS = 16
COLS = 30
TOTAL_MINES = 99

_ROOT_PATH = Path(__file__).resolve()
_CNN_MODEL_PATH = _ROOT_PATH.parent / "model" / "minesweeper_cnn.pth"
_CNN_META_PATH = _ROOT_PATH.parent / "model" / "minesweeper_meta.json"

if __name__ == "__main__":
    main(rows=ROWS, cols=COLS, total_mines=TOTAL_MINES, cnn_meta_path=_CNN_META_PATH, cnn_model_path=_CNN_MODEL_PATH)
