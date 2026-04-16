"""
Minesweeper UI 模块
提供终端打印和报告展示功能。
"""

from .terminal import (
    Colors,
    get_local_grid_str,
    print_board_matrix_for_debug,
    print_boxed_report,
    print_naming_guide,
)

__all__ = [
    "Colors",
    "get_local_grid_str",
    "print_board_matrix_for_debug",
    "print_boxed_report",
    "print_naming_guide",
]
