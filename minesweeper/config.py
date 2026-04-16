import os
from pathlib import Path

# ── 项目路径 ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEMPLATE_DIR = PROJECT_ROOT / "templates"
UNKNOWN_DIR = PROJECT_ROOT / "templates_unknown"

# ── 游戏参数（专家模式）──────────────────────────────────
ROWS = 16
COLS = 30
TOTAL_MINES = 99

# ── 视觉引擎参数 ─────────────────────────────────────────
CELL_SIZE = 64
TEMPLATE_SIZE = (64, 64)
MATCH_PAD = 6  # 模板匹配滑动窗口容差 (像素)

# ── 评分权重 ─────────────────────────────────────────────
TM_WEIGHT = 0.5  # 模板匹配 (TM_CCOEFF_NORMED) 权重
SSIM_WEIGHT = 0.5  # SSIM 结构相似度权重

# ── 识别阈值 ─────────────────────────────────────────────
NORMAL_THRESHOLD = 0.55
FORCE_THRESHOLD = 0.35
COLOR_FALLBACK_THRESHOLD = 0.25

# ── 并行扫描 ─────────────────────────────────────────────
MAX_SCAN_WORKERS = min(8, os.cpu_count() or 4)
