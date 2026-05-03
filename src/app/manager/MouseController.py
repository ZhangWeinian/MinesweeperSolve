import sys

import pyautogui
from pynput import keyboard

_DPI_SCALE = 1.0
if sys.platform == "win32":
    try:
        import ctypes

        _DPI_SCALE = ctypes.windll.shcore.GetScaleFactorForDevice(0) / 100.0
    except Exception:
        pass


class BotState:
    """全局状态对象，供键盘监听器和主线程共享"""

    __slots__ = ("stop", "waiting", "decision")

    def __init__(self):
        self.stop = False
        self.waiting = False
        self.decision: str | None = None


def start_keyboard_listener(state: BotState, on_extra_key=None):
    """启动一个独立线程监听键盘输入，更新 BotState 对象"""

    def on_key_press(key):
        if key == keyboard.Key.esc:
            state.stop = True
            import os

            os._exit(0)

        if state.waiting:
            if key == keyboard.Key.left:
                state.decision = "left"
            elif key == keyboard.Key.right:
                state.decision = "right"
            elif key == keyboard.Key.enter:
                state.decision = "enter"

        if on_extra_key is not None:
            on_extra_key(key, state)

    listener = keyboard.Listener(on_press=on_key_press)
    listener.start()

    return listener


def _physical_to_logical(phys_x: int, phys_y: int) -> tuple[int, int]:
    """将物理像素坐标转换为 Windows 逻辑坐标"""

    if abs(_DPI_SCALE - 1.0) < 1e-9:
        return phys_x, phys_y
    else:
        return int(phys_x / _DPI_SCALE), int(phys_y / _DPI_SCALE)


def click(x: int, y: int):
    """左键单击 (接受物理坐标，内部自动转换)"""

    log_x, log_y = _physical_to_logical(x, y)
    pyautogui.click(log_x, log_y)


def right_click(x: int, y: int):
    """右键单击 (接受物理坐标，内部自动转换)"""

    log_x, log_y = _physical_to_logical(x, y)
    pyautogui.rightClick(log_x, log_y)
