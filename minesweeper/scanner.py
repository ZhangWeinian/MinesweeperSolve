"""并行化棋盘扫描模块。

利用 ThreadPoolExecutor 按行并行识别（cv2 / numpy 操作会释放 GIL），
然后顺序处理副作用（UI 提示、文件写入、鼠标操作）。
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import pyautogui

from .config import COLS, MAX_SCAN_WORKERS, ROWS, UNKNOWN_DIR
from .ui import Colors as C
from .ui import print_naming_guide
from .vision import CellRecognizer, ScreenCapture, binarize_cell


def _recognize_row(recognizer, img_np, grid_map, row, cols, error_counts):
    """处理单行所有格子的识别（线程安全：仅读取 numpy + cv2 运算）。"""
    results = []
    for c in range(cols):
        cell_data = grid_map[row][c]
        y1, y2, x1, x2 = cell_data["slice"]
        cell_img = img_np[y1:y2, x1:x2].copy()  # 连续内存副本

        coord_key = f"{row}_{c}"
        force_guess = error_counts.get(coord_key, 0) >= 3
        result = recognizer.identify(cell_img, force_guess=force_guess)
        results.append((row, c, result, cell_img))
    return results


def scan_full_board(capture, recognizer, error_counts, state):
    """扫描完整棋盘。

    Args:
        capture: ScreenCapture 实例
        recognizer: CellRecognizer 实例
        error_counts: dict[str, int]，坐标 -> 连续未识别次数
        state: BotState 实例（提供 .stop / .waiting / .decision 属性）

    Returns:
        (board, stats) 或 (None, stats) 表示中断/需要重扫
    """
    if state.stop:
        os._exit(0)

    img_np = capture.grab_frame()
    rows, cols = capture.rows, capture.cols

    # ── 第一阶段：多线程并行识别 ────────────────────────
    all_results = []
    with ThreadPoolExecutor(max_workers=MAX_SCAN_WORKERS) as pool:
        futures = [
            pool.submit(
                _recognize_row,
                recognizer,
                img_np,
                capture.grid_map,
                r,
                cols,
                error_counts,
            )
            for r in range(rows)
        ]
        for f in futures:
            all_results.extend(f.result())

    # ── 第二阶段：顺序后处理（副作用：鼠标/文件/UI）────
    board = [[-1 for _ in range(cols)] for _ in range(rows)]
    stats = {"blind": 0, "blank": 0, "flag": 0, "number": 0}

    for r, c, result, cell_img in all_results:
        if state.stop:
            os._exit(0)

        coord_key = f"{r}_{c}"
        error_times = error_counts.get(coord_key, 0)

        if result == "Q":
            target_x = capture.grid_map[r][c]["cx"]
            target_y = capture.grid_map[r][c]["cy"]
            print(
                f"\n{C.YELLOW}🔄 侦测到误触的【？】标记于 ({r}, {c})，"
                f"停顿 0.8s 后执行右键修复...{C.RESET}"
            )
            time.sleep(0.8)
            pyautogui.rightClick(target_x, target_y)
            time.sleep(0.2)
            return None, stats

        if result == "?":
            error_counts[coord_key] = error_times + 1
            unknown_dir = UNKNOWN_DIR
            if not unknown_dir.exists():
                unknown_dir.mkdir(parents=True)

            for old_file in unknown_dir.iterdir():
                old_file.unlink()

            cv2.imwrite(str(unknown_dir / f"unknown_r{r}_c{c}_color.png"), cell_img)
            shape, _ = binarize_cell(cell_img)
            if shape is not None:
                cv2.imwrite(str(unknown_dir / f"unknown_r{r}_c{c}_bw.png"), shape)

            err_msg = [
                f"\n{C.RED}{C.BOLD}" + "!" * 50,
                f"⚠️ [视觉告警] 坐标 ({r}, {c}) 出现未知字形！"
                f"(历史报错: {error_times + 1}/3 次)",
                "!" * 50 + f"{C.RESET}",
            ]
            print("\n".join(err_msg))
            print_naming_guide()

            print(f"{C.MAGENTA}⏸️ 已暂停！{C.RESET}")
            print(
                f"👉 {C.YELLOW}若为新字形，放入模板后按键盘【回车】重载并继续。{C.RESET}"
            )

            state.waiting = True
            state.decision = None
            while state.waiting and state.decision != "enter" and not state.stop:
                time.sleep(0.05)
            state.waiting = False

            recognizer.load_templates()
            return None, stats

        if coord_key in error_counts:
            del error_counts[coord_key]

        board[r][c] = result
        if result == -1:
            stats["blind"] += 1
        elif result == 0:
            stats["blank"] += 1
        elif result == "F":
            stats["flag"] += 1
        else:
            stats["number"] += 1

    return board, stats
