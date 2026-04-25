import os
from concurrent.futures import ThreadPoolExecutor

from .config import MAX_SCAN_WORKERS


def _recognize_row(recognizer, img_np, grid_map, row, cols):
    """处理单行所有格子的识别（线程安全：仅读取 numpy + cv2 运算）。"""
    results = []
    for c in range(cols):
        cell_data = grid_map[row][c]
        y1, y2, x1, x2 = cell_data["slice"]
        cell_img = img_np[y1:y2, x1:x2].copy()  # 连续内存副本

        result = recognizer.identify(cell_img)
        results.append((row, c, result, cell_img))
    return results


def scan_full_board(capture, recognizer, state):
    """扫描完整棋盘。

    Args:
        capture: ScreenCapture 实例
        recognizer: CellRecognizer 实例
        state: BotState 实例（提供 .stop / .waiting / .decision 属性）

    Returns:
        (board, stats) 或 (None, stats) 表示中断/需要重扫
    """
    if state.stop:
        os._exit(0)

    img_np = capture.grab_frame()
    rows, cols = capture.rows, capture.cols

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
            )
            for r in range(rows)
        ]
        for f in futures:
            all_results.extend(f.result())

    board = [[-1 for _ in range(cols)] for _ in range(rows)]
    stats = {"blind": 0, "blank": 0, "flag": 0, "number": 0}

    for r, c, result, cell_img in all_results:
        if state.stop:
            os._exit(0)

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
