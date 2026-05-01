import ctypes
import os

import pyautogui

from minesweeper.BusController import run_auto_bot
from minesweeper.TerminalPrint import Colors as C


def main(rows, cols, total_mines, cnn_model_path, cnn_meta_path):
    """执行完全自动化的扫雷主循环，带有系统级故障安全保护。"""
    ctypes.windll.user32.SetProcessDPIAware()

    try:
        run_auto_bot(
            rows=rows,
            cols=cols,
            total_mines=total_mines,
            model_path=cnn_model_path,
            meta_path=cnn_meta_path,
        )
    except pyautogui.FailSafeException:
        print(f"\n{C.RED}🛑 触发物理紧急制动！自动驾驶中止。{C.RESET}")
        os._exit(0)
    except KeyboardInterrupt:
        print(f"\n{C.RED}🛑 接收到 Ctrl+C 中断信号！自动驾驶彻底中止。{C.RESET}")
        os._exit(0)


__all__ = ["main"]
