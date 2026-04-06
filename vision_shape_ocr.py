import ctypes
import os
import time
from typing import Any, List

import cv2
import mss
import numpy as np
from pynput import keyboard

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_physical_mouse_pos():
    class POINT(ctypes.Structure):
        _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

    pt = POINT()
    ctypes.windll.user32.GetPhysicalCursorPos(ctypes.byref(pt))
    return (pt.x, pt.y)


class HighResShapeOCR:
    def __init__(self, rows=16, cols=30):
        self.rows = rows
        self.cols = cols
        self.grid_map: List[List[Any]] = [
            [None for _ in range(cols)] for _ in range(rows)
        ]
        self.sct = mss.mss()
        self.template_dir = os.path.join(BASE_DIR, "templates")
        self.templates = {}
        self.CELL_SIZE = 64
        self.TEMPLATE_SIZE = (64, 64)
        self.offset_x = 0
        self.offset_y = 0
        self.monitor = None

        self.load_templates()

    def load_templates(self):
        self.templates.clear()
        if not os.path.exists(self.template_dir):
            os.makedirs(self.template_dir)

        shape_count = 0
        for filename in os.listdir(self.template_dir):
            if filename.lower().endswith(".png"):
                base_name = os.path.splitext(filename)[0]
                base_label = base_name.split("_")[0].upper()
                path = os.path.join(self.template_dir, filename)

                # 【新增】加入对 Q 的支持
                if base_label in ["1", "2", "3", "4", "5", "6", "7", "8", "F", "Q"]:
                    tpl_bw = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    if tpl_bw is not None:
                        self.templates.setdefault(base_label, []).append(
                            cv2.resize(tpl_bw, self.TEMPLATE_SIZE)
                        )
                        shape_count += 1

        print(f"✅ [视觉特征库] 成功挂载 {shape_count} 个超清 64x64 模板。")

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
            # 容差改为上限100，更稳定
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
        print("\n" + "=" * 60)
        print("🎯 【全景双锚点光学纠偏引擎】")
        print("👉 1. 将鼠标停留在【第一行、第一列】格子上，按 'F4'")
        print("👉 2. 将鼠标停留在【最后一行、最后一列】格子上，按 'F5'")

        pt1, pt2 = None, None

        def on_press(key):
            nonlocal pt1, pt2
            try:
                if key == keyboard.Key.f4 and pt1 is None:
                    pt1 = get_physical_mouse_pos()
                    print(f"✅ 捕获锚点一: {pt1}")
                elif key == keyboard.Key.f5 and pt1 is not None and pt2 is None:
                    pt2 = get_physical_mouse_pos()
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
        img_np = np.array(sct_img)[:, :, :3]
        self.offset_x, self.offset_y = self.monitor["left"], self.monitor["top"]

        c1_x, c1_y = self._find_true_center(img_np, pt1[0], pt1[1])
        c2_x, c2_y = self._find_true_center(img_np, pt2[0], pt2[1])

        step_x = (c2_x - c1_x) / (self.cols - 1)
        step_y = (c2_y - c1_y) / (self.rows - 1)

        print(f"📐 物理校准完毕 -> 步长 X: {step_x:.4f}px | 步长 Y: {step_y:.4f}px")

        half_s = self.CELL_SIZE // 2

        for r in range(self.rows):
            for c in range(self.cols):
                rel_cx = int(round(c1_x + c * step_x))
                rel_cy = int(round(c1_y + r * step_y))

                abs_x = rel_cx + self.offset_x
                abs_y = rel_cy + self.offset_y

                x_start = rel_cx - half_s
                y_start = rel_cy - half_s
                x_end = rel_cx + half_s
                y_end = rel_cy + half_s

                self.grid_map[r][c] = {
                    "cx": abs_x,
                    "cy": abs_y,
                    "slice": (y_start, y_end, x_start, x_end),
                }

    def binarize_cell(self, cell_64x64_img):
        b, g, r_ch = np.median(cell_64x64_img, axis=(0, 1))
        is_opened = int(b) - int(r_ch) < 25

        if not is_opened:
            # 蓝底找红色的旗帜
            hsv = cv2.cvtColor(cell_64x64_img, cv2.COLOR_BGR2HSV)
            mask1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
            mask2 = cv2.inRange(
                hsv, np.array([170, 100, 100]), np.array([180, 255, 255])
            )
            red_mask = cv2.bitwise_or(mask1, mask2)

            # 红旗面积阈值从 15 提高到 50，防止红色噪点干扰
            if np.count_nonzero(red_mask) > 50:
                return red_mask, False

            # 蓝底找黑/深色的问号
            gray = cv2.cvtColor(cell_64x64_img, cv2.COLOR_BGR2GRAY)
            bg_level = np.median(gray)
            _, dark_mask = cv2.threshold(
                gray, max(0, bg_level - 30), 255, cv2.THRESH_BINARY_INV
            )
            # 深色面积阈值从 20 提高到 100，过滤掉 3D 边框的阴影噪点！
            if np.count_nonzero(dark_mask) > 100:
                return dark_mask, False

            # 如果暗斑小于 100 像素，坚决认为是纯盲区，返回 None
            return None, False

        # 3. 白底找数字
        gray = cv2.cvtColor(cell_64x64_img, cv2.COLOR_BGR2GRAY)

        if np.min(gray) > 130:
            return None, True

        bg_level = np.median(gray)
        _, binary = cv2.threshold(
            gray, max(0, bg_level - 30), 255, cv2.THRESH_BINARY_INV
        )

        # 【优化】数字面积阈值从 25 提高到 80，过滤掉空地上的微小划痕/噪点！
        if np.count_nonzero(binary) < 80:
            return None, True

        return binary, True

    def identify_cell(self, cell_64x64_img, force_guess=False):
        shape, is_opened = self.binarize_cell(cell_64x64_img)

        if shape is None:
            return 0 if is_opened else -1

        best_match, max_score = None, -1
        for label, tpl_list in self.templates.items():
            for tpl in tpl_list:
                res = cv2.matchTemplate(shape, tpl, cv2.TM_CCOEFF_NORMED)
                score = res[0][0]
                if score > max_score:
                    max_score = score
                    best_match = label

        threshold = 0.45 if force_guess else 0.7

        if max_score > threshold and best_match is not None:
            return best_match if best_match in ["F", "Q"] else int(best_match)
        else:
            return 0 if force_guess else "?"
