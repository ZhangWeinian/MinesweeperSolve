import ctypes
import os

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


def find_true_center(sct_img, mouse_x, mouse_y, monitor_offset_x, monitor_offset_y):
    """
    【光学纠偏核心】：在鼠标点击位置周围 150x150 像素内，
    寻找最靠近中心的蓝色/灰色格子的真实物理中心点！
    """
    rel_x = mouse_x - monitor_offset_x
    rel_y = mouse_y - monitor_offset_y

    # 截取鼠标周围一小块区域 (防越界)
    h_img, w_img = sct_img.shape[:2]
    search_radius = 75

    x_start = max(0, rel_x - search_radius)
    y_start = max(0, rel_y - search_radius)
    x_end = min(w_img, rel_x + search_radius)
    y_end = min(h_img, rel_y + search_radius)

    roi = sct_img[y_start:y_end, x_start:x_end]

    # 转 HSV 找盖子(蓝色)或空地(灰白)的边缘
    # 扫雷的格子之间有一条深色的网格线，我们通过 Canny 边缘检测找出那个四方形！
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # 寻找闭合的四边形轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    best_center = None
    min_dist = float("inf")

    roi_center = (search_radius, search_radius)

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # 你的格子大约是 90x90，我们找宽和高在 75~100 之间的轮廓
        if 75 < w < 100 and 75 < h < 100:
            cx = x + w // 2
            cy = y + h // 2
            # 找到离鼠标点击位置最近的那个格子的中心
            dist = (cx - roi_center[0]) ** 2 + (cy - roi_center[1]) ** 2
            if dist < min_dist:
                min_dist = dist
                # 转换回全图相对坐标
                best_center = (x_start + cx, y_start + cy)

    if best_center is None:
        print("\n❌ 光学纠偏失败！未在鼠标周围找到大小约为 90x90 的格子边缘！")
        print("退回使用鼠标点击的原始坐标...")
        return (rel_x, rel_y)

    return best_center


def run_grid_test():
    print("=" * 60)
    print("🎯 【30x16 光学纠偏网格测试 (双屏抗震版)】")
    print("=" * 60)

    top_left_raw = None
    bottom_right_raw = None

    def on_press(key):
        nonlocal top_left_raw, bottom_right_raw
        try:
            if key == keyboard.Key.f4 and top_left_raw is None:
                top_left_raw = get_physical_mouse_pos()
                print(f"✅ 捕获起点大致坐标: {top_left_raw}")
            elif (
                key == keyboard.Key.f5
                and top_left_raw is not None
                and bottom_right_raw is None
            ):
                bottom_right_raw = get_physical_mouse_pos()
                print(f"✅ 捕获终点大致坐标: {bottom_right_raw}")
                return False
        except AttributeError:
            pass

    print("👉 1. 将鼠标随意丢在【第一行第一列】格子内（无需精准中心），按 'F4'")
    print("👉 2. 将鼠标随意丢在【最后一行最后一列】格子内，按 'F5'")

    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

    print("\n📸 全局虚拟屏幕抓取中 (支持多屏穿透)...")

    # 【多屏大绝杀】：抓取所有屏幕组成的虚拟大桌面的所有像素
    with mss.mss() as sct:
        monitor = sct.monitors[0]
        sct_img = sct.grab(monitor)
        img_np = np.array(sct_img)[:, :, :3]
        offset_x, offset_y = monitor["left"], monitor["top"]

    print("🔍 启动 OpenCV 光学中心纠偏引擎...")

    # 分别找出真正的、像素级完美的圆心坐标！
    c1_rel_x, c1_rel_y = find_true_center(
        img_np, top_left_raw[0], top_left_raw[1], offset_x, offset_y
    )
    c2_rel_x, c2_rel_y = find_true_center(
        img_np, bottom_right_raw[0], bottom_right_raw[1], offset_x, offset_y
    )

    print(f"🎯 光学校准完毕！")
    print(f"   原点真实相对坐标: ({c1_rel_x}, {c1_rel_y})")
    print(f"   终点真实相对坐标: ({c2_rel_x}, {c2_rel_y})")

    # ---------------------------------------------------------
    # 根据光学找出的首尾圆心，动态计算绝对浮点步长
    # ---------------------------------------------------------
    step_x = (c2_rel_x - c1_rel_x) / 29.0
    step_y = (c2_rel_y - c1_rel_y) / 15.0

    print(f"📐 动态计算物理步长 -> X轴: {step_x:.4f}px | Y轴: {step_y:.4f}px")

    cell_size = 86
    half_size = cell_size // 2
    annotated_img = img_np.copy()

    test_dir = os.path.join(BASE_DIR, "test_slices")
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    for f in os.listdir(test_dir):
        os.remove(os.path.join(test_dir, f))

    corners = [(0, 0), (0, 29), (15, 0), (15, 29), (7, 15)]

    for r in range(16):
        for c in range(30):
            # 浮点累加后精确取整
            rel_cx = int(round(c1_rel_x + c * step_x))
            rel_cy = int(round(c1_rel_y + r * step_y))

            x_start = rel_cx - half_size
            y_start = rel_cy - half_size
            x_end = rel_cx + half_size
            y_end = rel_cy + half_size

            cv2.rectangle(
                annotated_img, (x_start, y_start), (x_end, y_end), (0, 0, 255), 2
            )
            cv2.drawMarker(
                annotated_img,
                (rel_cx, rel_cy),
                (0, 255, 0),
                markerType=cv2.MARKER_CROSS,
                markerSize=10,
                thickness=1,
            )

            if (r, c) in corners:
                slice_img = img_np[y_start:y_end, x_start:x_end]
                cv2.imwrite(os.path.join(test_dir, f"check_r{r}_c{c}.png"), slice_img)

    preview_path = os.path.join(BASE_DIR, "grid_preview_4k_v2.png")
    cv2.imwrite(preview_path, annotated_img)

    print("\n" + "=" * 60)
    print("🎉 光学纠偏测试完成！")
    print(f"1. 🌐 全景网格已保存至: {preview_path}")
    print(f"2. 🔲 角落切片已保存至: {test_dir}")
    print("=" * 60)


if __name__ == "__main__":
    ctypes.windll.user32.SetProcessDPIAware()
    run_grid_test()
