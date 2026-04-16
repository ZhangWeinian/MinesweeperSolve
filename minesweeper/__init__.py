"""
Minesweeper 包根入口

在此对外暴露应用核心运行器等功能。
"""

import ctypes
import os

import pyautogui

from .bot import run_auto_bot
from .ui import Colors as C


def main():
    """执行完全自动化的扫雷主循环，带有系统级故障安全保护。"""
    ctypes.windll.user32.SetProcessDPIAware()

    try:
        run_auto_bot()
    except pyautogui.FailSafeException:
        print(f"\n{C.RED}🛑 触发物理紧急制动！自动驾驶中止。{C.RESET}")
        os._exit(0)
    except KeyboardInterrupt:
        print(f"\n{C.RED}🛑 接收到 Ctrl+C 中断信号！自动驾驶彻底中止。{C.RESET}")
        os._exit(0)


__all__ = ["main", "run_auto_bot"]
