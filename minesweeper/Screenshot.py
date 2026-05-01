import ctypes
from typing import Any

import cv2
import mss
import numpy as np
from pynput import keyboard


def _get_physical_mouse_pos():
    """通过 Win32 API 获取物理鼠标坐标（绕过 DPI 缩放）。"""

    class POINT(ctypes.Structure):
        _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

    pt = POINT()
    ctypes.windll.user32.GetPhysicalCursorPos(ctypes.byref(pt))
    return (pt.x, pt.y)


class ScreenCapture:
    """屏幕截图 + 双锚点校准 + 网格坐标映射。"""

    def __init__(self, rows: int, cols: int, cell_size: int = 64):
        self.rows = rows
        self.cols = cols
        self.cell_size = cell_size
        self.grid_map: list[list[Any]] = [[None for _ in range(cols)] for _ in range(rows)]
        self.sct = mss.mss()
        self.offset_x = 0
        self.offset_y = 0
        self.monitor = self.sct.monitors[0]

    def grab_frame(self):
        """截取当前屏幕帧，返回 BGR numpy 数组。"""
        sct_img = self.sct.grab(self.monitor)
        return np.array(sct_img)[:, :, :3]

    def _find_true_center(self, sct_img, raw_x, raw_y):
        rel_x = raw_x - self.offset_x
        rel_y = raw_y - self.offset_y
        h_img, w_img = sct_img.shape[:2]

        search_r = 75
        x_start = max(0, rel_x - search_r)
        y_start = max(0, rel_y - search_r)
        x_end = min(w_img, rel_x + search_r)
        y_end = min(h_img, rel_y + search_r)

        roi = sct_img[y_start:y_end, x_start:x_end]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        best_center = None
        min_dist = float("inf")
        roi_center = (search_r, search_r)

        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if 75 < w < 100 and 75 < h < 100:
                cx = x + w // 2
                cy = y + h // 2
                dist = (cx - roi_center[0]) ** 2 + (cy - roi_center[1]) ** 2
                if dist < min_dist:
                    min_dist = dist
                    best_center = (x_start + cx, y_start + cy)

        if best_center is None:
            print("\n❌ 光学纠偏失败！使用原始鼠标坐标...")
            return (rel_x, rel_y)
        return best_center

    def calibrate(self):
        """双锚点校准：用户点击左上角和右下角格子来建立网格映射。"""
        print("\n" + "=" * 60)
        print("🎯 【全景双锚点光学纠偏引擎】")
        print("👉 1. 将鼠标停留在【第一行、第一列】格子上，按 'F4'")
        print("👉 2. 将鼠标停留在【最后一行、最后一列】格子上，按 'F5'")

        pt1, pt2 = None, None

        def on_press(key):
            nonlocal pt1, pt2
            try:
                if key == keyboard.Key.f4 and pt1 is None:
                    pt1 = _get_physical_mouse_pos()
                    print(f"✅ 捕获锚点一: {pt1}")
                elif key == keyboard.Key.f5 and pt1 is not None and pt2 is None:
                    pt2 = _get_physical_mouse_pos()
                    print(f"✅ 捕获锚点二: {pt2}")
                    return False
            except AttributeError:
                pass

        with keyboard.Listener(on_press=on_press) as listener:  # type: ignore
            listener.join()

        assert pt1 is not None and pt2 is not None, "锚点坐标丢失异常"

        print("\n📸 全局虚拟屏幕抓取中 (支持多屏穿透)...")
        self.monitor = self.sct.monitors[0]
        sct_img = self.sct.grab(self.monitor)
        img_np = cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGRA2BGR)
        self.offset_x, self.offset_y = self.monitor["left"], self.monitor["top"]

        c1_x, c1_y = self._find_true_center(img_np, pt1[0], pt1[1])
        c2_x, c2_y = self._find_true_center(img_np, pt2[0], pt2[1])

        step_x = (c2_x - c1_x) / (self.cols - 1)
        step_y = (c2_y - c1_y) / (self.rows - 1)

        print(f"📐 物理校准完毕 -> 步长 X: {step_x:.4f}px | 步长 Y: {step_y:.4f}px")

        half_s = self.cell_size // 2

        for r in range(self.rows):
            for c in range(self.cols):
                rel_cx = int(round(c1_x + c * step_x))
                rel_cy = int(round(c1_y + r * step_y))

                abs_x = rel_cx + self.offset_x
                abs_y = rel_cy + self.offset_y

                self.grid_map[r][c] = {
                    "cx": abs_x,
                    "cy": abs_y,
                    "slice": (
                        rel_cy - half_s,
                        rel_cy + half_s,
                        rel_cx - half_s,
                        rel_cx + half_s,
                    ),
                }

    def get_cell_center(self, r: int, c: int) -> tuple[int, int] | None:
        """返回格子 (r, c) 的屏幕中心坐标 (x, y)；未校准时返回 None。"""
        cell = self.grid_map[r][c]
        if cell is None:
            return None
        return cell["cx"], cell["cy"]

    def get_cell_image(self, frame, r: int, c: int):
        """从截图帧中裁切出格子 (r, c) 的图像副本（BGR numpy 数组）。"""
        y1, y2, x1, x2 = self.grid_map[r][c]["slice"]
        return frame[y1:y2, x1:x2].copy()
