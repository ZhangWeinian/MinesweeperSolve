import threading


class ConsistencyChecker:
    """跨帧一致性校验器，防止图像抖动导致误识别污染求解器"""

    def __init__(self):
        self._lock = threading.Lock()
        self._history: dict[tuple, list] = {}
        self._is_first_frame = True

    def reset_history(self):
        """用户介入操作时调用，清空记忆，重新开始"""
        with self._lock:
            self._history.clear()
            self._is_first_frame = True

    def check(self, pos: tuple, value, confidence: float):
        """
        返回: (final_value, is_stable)
        - is_stable=True: 识别稳定，请更新棋盘
        - is_stable=False: 识别波动，请忽略本次结果，保留上一帧旧值！
        """
        # 盲区和空白由 OpenCV 色彩决定，非常稳定，不参与校验
        if value == -1 or value == 0:
            return value, True

        with self._lock:
            if pos not in self._history:
                self._history[pos] = []

            self._history[pos].append((value, confidence))

            # 只保留最近3帧
            if len(self._history[pos]) > 3:
                self._history[pos].pop(0)

            hist = self._history[pos]

            # 第一帧强制采信（因为此时没有旧值可保留）
            if self._is_first_frame:
                return value, True

            # 策略：最近2帧一致则可信
            if len(hist) >= 2:
                if hist[-1][0] == hist[-2][0]:
                    # 稳定了，清空该点历史，减少内存占用
                    self._history[pos] = [hist[-1]]
                    return hist[-1][0], True
                else:
                    # 2帧不一致，看是否有第3帧
                    if len(hist) == 3:
                        # 取3帧中置信度最高的作为最终结果
                        best_val = max(hist, key=lambda x: x[1])[0]
                        self._history[pos] = [(best_val, 1.0)]
                        return best_val, True
                    else:
                        # 还在等待第3帧，返回不稳定标记
                        return None, False

            # 只有1帧历史，且不是第一帧，返回不稳定标记
            return None, False
