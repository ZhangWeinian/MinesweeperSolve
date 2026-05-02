import sys

if sys.platform == "win32":
    import ctypes
    import os

    import pyautogui

    from src.app.manager.BusController import run_auto_bot
    from src.app.manager.TerminalPrint import Colors as C

    def main(rows, cols, total_mines, cnn_model_path, cnn_meta_path, root_path=None):
        """执行完全自动化的扫雷主循环，带有系统级故障安全保护。"""
        ctypes.windll.user32.SetProcessDPIAware()

        try:
            run_auto_bot(
                rows=rows,
                cols=cols,
                total_mines=total_mines,
                model_path=cnn_model_path,
                meta_path=cnn_meta_path,
                root_path=root_path,
            )
        except pyautogui.FailSafeException:
            print(f"\n{C.RED}🛑 触发物理紧急制动！自动驾驶中止。{C.RESET}")
            os._exit(0)
        except KeyboardInterrupt:
            print(f"\n{C.RED}🛑 接收到 Ctrl+C 中断信号！自动驾驶彻底中止。{C.RESET}")
            os._exit(0)

else:

    def main(*args, **kwargs):
        """非 Windows 环境，此函数为空实现，仅用于规避 IDE 静态检查报错"""
        pass


__all__ = ["main"]
