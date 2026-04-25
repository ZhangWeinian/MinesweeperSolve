import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_ROOT / "model"
CNN_MODEL_PATH = MODEL_DIR / "minesweeper_cnn.pth"
CNN_META_PATH = MODEL_DIR / "minesweeper_meta.json"

ROWS = 16
COLS = 30
TOTAL_MINES = 99

CELL_SIZE = 64

MAX_SCAN_WORKERS = min(8, os.cpu_count() or 4)
