import pyautogui
from pynput import keyboard


class BotState:
    """线程间共享的轻量状态对象。"""

    __slots__ = ("stop", "waiting", "decision")

    def __init__(self):
        self.stop = False
        self.waiting = False
        self.decision: str | None = None


def start_keyboard_listener(state: BotState, on_extra_key=None):
    """启动键盘监听器并返回 Listener 实例。"""

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


def click(x: int, y: int):
    """左键单击指定屏幕坐标。"""

    pyautogui.click(x, y)


def right_click(x: int, y: int):
    """右键单击指定屏幕坐标。"""

    pyautogui.rightClick(x, y)
