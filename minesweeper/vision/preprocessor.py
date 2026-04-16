import cv2
import numpy as np

# ── 全局缓存的形态学内核（避免每次调用重新创建）──────────
KERNEL_3x3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


def binarize_cell(cell_img):
    """将 64×64 BGR 格子图像二值化。

    Returns:
        (binary_mask | None, is_opened: bool)
        - binary_mask: 前景掩码 (uint8)，None 表示空白或未翻开
        - is_opened: True = 已翻开格
    """
    hsv = cv2.cvtColor(cell_img, cv2.COLOR_BGR2HSV)
    center_hsv = hsv[12:52, 12:52]
    median_s = np.median(center_hsv[:, :, 1])
    is_opened = median_s < 60

    if not is_opened:
        # 红色旗帜检测
        mask1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([170, 100, 100]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_or(mask1, mask2)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, KERNEL_3x3)

        if np.count_nonzero(red_mask) > 50:
            return red_mask, False

        # 暗色特征检测（未翻开格上的问号等）
        gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
        bg_level = np.median(gray)
        _, dark_mask = cv2.threshold(
            gray, max(0, bg_level - 30), 255, cv2.THRESH_BINARY_INV
        )
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, KERNEL_3x3)

        if np.count_nonzero(dark_mask) > 100:
            return dark_mask, False

        return None, False

    else:
        gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)

        # 中心区域对比度检查 → 过滤纯空白格
        center_gray = gray[8:56, 8:56]
        if np.ptp(center_gray) < 35:
            return None, True

        # Gaussian 模糊 + Otsu 自适应阈值
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        _, binary = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # 形态学开运算：消除微小噪点
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, KERNEL_3x3)

        if np.count_nonzero(binary) < 80:
            return None, True

        return binary, True


def get_digit_color_label(cell_bgr, binary_mask):
    """利用数字像素的 HSV 颜色辅助识别。

    标准扫雷配色:
      1=蓝  2=绿  3=红  4=深蓝  5=暗红  6=青  7=黑  8=灰

    Returns:
        label 字符串 ("1"-"8") 或 None
    """
    if binary_mask is None:
        return None

    expanded = cv2.dilate(binary_mask, KERNEL_3x3, iterations=1)
    hsv = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2HSV)
    digit_pixels = hsv[expanded > 0]

    if len(digit_pixels) < 20:
        return None

    avg_h = float(np.median(digit_pixels[:, 0]))
    avg_s = float(np.median(digit_pixels[:, 1]))
    avg_v = float(np.median(digit_pixels[:, 2]))

    # 低饱和度 → 无彩色 (7=黑, 8=灰)
    if avg_s < 40:
        return "7" if avg_v < 100 else "8"

    # 蓝色区间 (H ≈ 95-125)
    if 95 <= avg_h <= 125:
        return "1" if avg_v > 140 else "4"

    # 绿色区间 (H ≈ 35-85)
    if 35 <= avg_h <= 85:
        return "2"

    # 红色区间 (H ≈ 0-10 or 170-180)
    if avg_h <= 10 or avg_h >= 170:
        return "3" if avg_v > 140 else "5"

    # 青色区间 (H ≈ 80-95)
    if 80 <= avg_h < 95:
        return "6"

    return None
