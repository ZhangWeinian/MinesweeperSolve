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


def scan_full_board(capture, recognizer, state, collector=None):
    """扫描完整棋盘。

    Args:
        capture: ScreenCapture 实例
        recognizer: CellRecognizer 实例
        state: BotState 实例（提供 .stop / .waiting / .decision 属性）
        collector: DatasetCollector 实例（可选），用于保存训练样本

    Returns:
        (board, stats, cell_images)
        - board: 识别结果矩阵
        - stats: 统计字典
        - cell_images: dict[(row, col) -> ndarray]，仅含已识别的数字/旗帜格子图像，
          供后续误识别报告使用
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
    # 仅保留数字/旗帜格子的原始图像，供误识别报告查询
    cell_images: dict[tuple, object] = {}

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
            cell_images[(r, c)] = cell_img
            if collector is not None:
                collector.try_save_train(cell_img, "F", (r, c))
        else:
            stats["number"] += 1
            cell_images[(r, c)] = cell_img
            if collector is not None:
                collector.try_save_train(cell_img, result, (r, c))

    return board, stats, cell_images
