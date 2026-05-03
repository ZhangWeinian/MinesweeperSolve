import math
import sys
import time
from collections import defaultdict
from functools import lru_cache
from typing import Any

import numpy as np
from numba import njit


@lru_cache(maxsize=None)
def fast_comb(n, k):
    if k < 0 or k > n:
        return 0
    else:
        return math.comb(n, k)


@njit(fastmath=True, cache=True)
def _is_value_forbidden(c_idx, val, cell_to_eqs, cell_eq_counts, eq_targets, eq_unassigned, eq_current_mines):
    """约束传播剪枝：判断当前值是否被方程强制定死"""

    for k in range(cell_eq_counts[c_idx]):
        eq_idx = cell_to_eqs[c_idx, k]
        rem = eq_targets[eq_idx] - eq_current_mines[eq_idx]
        unassigned = eq_unassigned[eq_idx]

        if rem == 0 and val == 1:
            return True  # 雷已满，不能再放雷
        if rem == unassigned and val == 0:
            return True  # 剩余格子必须全是雷

    return False


@njit(fastmath=True, cache=True)
def _record_config(
    num_cells, current_assignment, current_reveal_mines, config_counts, cell_mine_counts, cell_reveal_dists
):
    """记录一个合法的全局配置，更新统计数据结构"""

    total_mines = 0
    for i in range(num_cells):
        if current_assignment[i] == 1:
            total_mines += 1

    for i in range(num_cells):
        if current_assignment[i] == 1:
            cell_mine_counts[total_mines, i] += 1
        else:
            local_mines = current_reveal_mines[i]
            cell_reveal_dists[total_mines, i, local_mines] += 1

    config_counts[total_mines] += 1


@njit(fastmath=True, cache=True)
def _update_eq_state(c_idx, val, cell_to_eqs, cell_eq_counts, eq_targets, eq_unassigned, eq_current_mines):
    """约束传播：在尝试一个赋值后，更新相关方程的状态，并检查是否产生矛盾"""

    is_valid = True
    for k in range(cell_eq_counts[c_idx]):
        eq_idx = cell_to_eqs[c_idx, k]
        eq_unassigned[eq_idx] -= 1

        if val == 1:
            eq_current_mines[eq_idx] += 1
        if eq_current_mines[eq_idx] > eq_targets[eq_idx]:
            is_valid = False
        if eq_current_mines[eq_idx] + eq_unassigned[eq_idx] < eq_targets[eq_idx]:
            is_valid = False

    return is_valid


@njit(fastmath=True, cache=True)
def _revert_eq_state(c_idx, val, cell_to_eqs, cell_eq_counts, eq_unassigned, eq_current_mines):
    """约束传播：在回溯时，恢复相关方程的状态"""
    for k in range(cell_eq_counts[c_idx]):
        eq_idx = cell_to_eqs[c_idx, k]
        eq_unassigned[eq_idx] += 1
        if val == 1:
            eq_current_mines[eq_idx] -= 1


@njit(fastmath=True, cache=True)
def _update_reveal_mines(c_idx, delta, num_cells, cell_neighbors_matrix, current_reveal_mines):
    """在尝试一个赋值后，更新相关格子的周围雷数统计"""

    if delta != 0:
        for j in range(num_cells):
            if cell_neighbors_matrix[c_idx, j] == 1:
                current_reveal_mines[j] += delta


@njit(fastmath=True, cache=True)
def dfs_numba(
    c_idx,
    num_cells,
    cell_to_eqs,
    cell_eq_counts,
    eq_targets,
    cell_neighbors_matrix,
    eq_unassigned,
    eq_current_mines,
    current_assignment,
    current_reveal_mines,
    config_counts,
    cell_mine_counts,
    cell_reveal_dists,
):
    """核心枚举函数，使用 Numba 加速"""

    if c_idx == num_cells:
        _record_config(
            num_cells, current_assignment, current_reveal_mines, config_counts, cell_mine_counts, cell_reveal_dists
        )
        return

    for val in range(2):
        if _is_value_forbidden(c_idx, val, cell_to_eqs, cell_eq_counts, eq_targets, eq_unassigned, eq_current_mines):
            continue

        current_assignment[c_idx] = val
        is_valid = _update_eq_state(
            c_idx, val, cell_to_eqs, cell_eq_counts, eq_targets, eq_unassigned, eq_current_mines
        )

        if is_valid:
            delta = 1 if val == 1 else 0
            _update_reveal_mines(c_idx, delta, num_cells, cell_neighbors_matrix, current_reveal_mines)
            dfs_numba(
                c_idx + 1,
                num_cells,
                cell_to_eqs,
                cell_eq_counts,
                eq_targets,
                cell_neighbors_matrix,
                eq_unassigned,
                eq_current_mines,
                current_assignment,
                current_reveal_mines,
                config_counts,
                cell_mine_counts,
                cell_reveal_dists,
            )
            _update_reveal_mines(c_idx, -delta, num_cells, cell_neighbors_matrix, current_reveal_mines)

        _revert_eq_state(c_idx, val, cell_to_eqs, cell_eq_counts, eq_unassigned, eq_current_mines)

    current_assignment[c_idx] = -1


class ExpertMinesweeperSolver:
    """基于全局枚举的数学求解器，适用于中后期复杂局面的精确分析"""

    def __init__(self, rows=16, cols=30, total_mines=99):
        self.rows = rows
        self.cols = cols
        self.total_mines = total_mines
        self._is_jitted = False
        self.suspicious_cells: list[tuple] = []
        self.neighbor_map = {}
        for r in range(self.rows):
            for c in range(self.cols):
                neighbors = []
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < self.rows and 0 <= nc < self.cols:
                            neighbors.append((nr, nc))
                self.neighbor_map[(r, c)] = neighbors

    def get_neighbors(self, r, c):
        """获取一个格子的邻居列表，预先计算以加速访问"""

        return self.neighbor_map[(r, c)]

    def _collect_supporting_clues(self, board, cell):
        """收集支持当前格子的线索"""

        clues = []
        for nr, nc in self.get_neighbors(*cell):
            val = board[nr][nc]
            if not (isinstance(val, int) and val > 0):
                continue
            unknowns, local_flags = [], 0

            for ar, ac in self.get_neighbors(nr, nc):
                if board[ar][ac] == -1:
                    unknowns.append((ar, ac))
                elif board[ar][ac] == "F":
                    local_flags += 1

            if cell in unknowns:
                clues.append(
                    {
                        "cell": (nr, nc),
                        "value": val,
                        "flags": local_flags,
                        "unknowns": len(unknowns),
                        "target": val - local_flags,
                    }
                )
        clues.sort(key=lambda item: (item["cell"][0], item["cell"][1]))

        return clues

    def _validate_cell_neighbors(self, r, c, val, local_flags, unknowns):
        """基于局部约束的快速异常检测，防止识别错误导致求解器崩溃"""

        if local_flags > val or val - local_flags > len(unknowns):
            print(f"⚠️  [识别警告] ({r},{c}) 数据异常，已跳过", file=sys.stderr)
            self.suspicious_cells.append((r, c, val))
            return False
        else:
            return True

    def _process_board_cell(self, board, r, c, equations, frontier_cells, trivial_safe, trivial_mine, unknown_cells):
        """处理单个格子，更新方程和边界格子集合"""

        val = board[r][c]
        if val == "F" or val == -1 or not (isinstance(val, int) and val > 0):
            if val == -1:
                unknown_cells.add((r, c))
            return

        unknowns, local_flags = [], 0
        for nr, nc in self.get_neighbors(r, c):
            if board[nr][nc] == -1:
                unknowns.append((nr, nc))
            elif board[nr][nc] == "F":
                local_flags += 1

        if not unknowns or not self._validate_cell_neighbors(r, c, val, local_flags, unknowns):
            return

        target = val - local_flags
        if target == 0:
            trivial_safe.update(unknowns)
        elif target == len(unknowns):
            trivial_mine.update(unknowns)
        else:
            frontier_cells.update(unknowns)
            equations.append({"cells": set(unknowns), "target": target})

    def _merge_dp(self, dp1, dp2, max_mines):
        """合并两个动态规划表"""

        merged = defaultdict(int)
        for m1, w1 in dp1.items():
            for m2, w2 in dp2.items():
                total = m1 + m2
                if total <= max_mines:
                    merged[total] += w1 * w2

        return dict(merged)

    def solve_step(self, board):
        """求解当前盘面，返回推荐的操作列表"""

        if not self._is_jitted:
            dummy_args = (
                np.zeros((1, 8), dtype=np.int32),
                np.zeros(1, dtype=np.int32),
                np.array([0], dtype=np.int8),
                np.array([[0]], dtype=np.int8),
                np.array([1], dtype=np.int8),
                np.array([0], dtype=np.int8),
                np.full(1, -1, dtype=np.int8),
                np.zeros(1, dtype=np.int8),
                np.zeros(2, dtype=np.int64),
                np.zeros((2, 1), dtype=np.int64),
                np.zeros((2, 1, 9), dtype=np.int64),
            )
            dfs_numba(0, 1, *dummy_args)
            self._is_jitted = True

        t_start = time.perf_counter()
        self.suspicious_cells = []
        equations, frontier_cells, unknown_cells = [], set(), set()
        trivial_safe, trivial_mine = set(), set()

        flag_count = sum(1 for r in range(self.rows) for c in range(self.cols) if board[r][c] == "F")
        for r in range(self.rows):
            for c in range(self.cols):
                self._process_board_cell(
                    board, r, c, equations, frontier_cells, trivial_safe, trivial_mine, unknown_cells
                )

        remaining_mines = self.total_mines - flag_count
        if remaining_mines < 0:
            raise ValueError("标记雷数超出上限！")

        baseline_prob = remaining_mines / max(1, len(unknown_cells))
        if trivial_safe or trivial_mine:
            calc_time_ms = (time.perf_counter() - t_start) * 1000
            details = {
                "blocks": 0,
                "universes": 1,
                "safe_found": len(trivial_safe),
                "mine_found": len(trivial_mine),
                "baseline_prob": baseline_prob,
                "top_candidates": [],
            }
            decisions = []

            for cell in trivial_mine:
                decisions.append(
                    {
                        "action": "FLAG",
                        "cell": cell,
                        "prob": 1.0,
                        "info_gain": 0.0,
                        "calc_time": calc_time_ms,
                        "details": details,
                        "debug": self._make_debug_info(board, cell, "平凡规则-必雷", 1.0, 0.0),
                    }
                )

            for cell in trivial_safe:
                decisions.append(
                    {
                        "action": "CLICK",
                        "cell": cell,
                        "prob": 0.0,
                        "info_gain": 0.0,
                        "calc_time": calc_time_ms,
                        "details": details,
                        "debug": self._make_debug_info(board, cell, "平凡规则-必安全", 0.0, 0.0),
                    }
                )

            for idx, d in enumerate(decisions):
                d["batch_info"] = (idx + 1, len(decisions))

            return decisions

        cell_to_eq_indices = defaultdict(list)
        for i, eq in enumerate(equations):
            for cell in eq["cells"]:
                cell_to_eq_indices[cell].append(i)

        blocks, eq_used = [], [False] * len(equations)
        for i in range(len(equations)):
            if eq_used[i]:
                continue

            curr_eqs, curr_cells, queue = [], set(), [i]
            eq_used[i] = True
            while queue:
                curr_idx = queue.pop(0)
                curr_eqs.append(equations[curr_idx])
                for c in equations[curr_idx]["cells"]:
                    if c not in curr_cells:
                        curr_cells.add(c)
                        for n_idx in cell_to_eq_indices[c]:
                            if not eq_used[n_idx]:
                                eq_used[n_idx] = True
                                queue.append(n_idx)

            blocks.append({"cells": list(curr_cells), "equations": curr_eqs})

        isolated_cells = unknown_cells - frontier_cells
        block_solutions = [self._compute_block_solution(block) for block in blocks]

        simple_block_dps = [{m: d["config_count"] for m, d in sol.items()} for sol in block_solutions]

        num_blocks = len(block_solutions)
        prefix_dp = [{} for _ in range(num_blocks + 1)]
        prefix_dp[0] = {0: 1}
        for i in range(num_blocks):
            prefix_dp[i+1] = self._merge_dp(prefix_dp[i], simple_block_dps[i], remaining_mines)

        suffix_dp = [{} for _ in range(num_blocks + 1)]
        suffix_dp[num_blocks] = {0: 1}
        for i in range(num_blocks - 1, -1, -1):
            suffix_dp[i] = self._merge_dp(suffix_dp[i+1], simple_block_dps[i], remaining_mines)

        other_dps = [self._merge_dp(prefix_dp[i], suffix_dp[i+1], remaining_mines) for i in range(num_blocks)]
        all_blocks_dp = prefix_dp[num_blocks]

        total_global_configs = sum(
            w * fast_comb(len(isolated_cells), remaining_mines - m)
            for m, w in all_blocks_dp.items()
            if 0 <= remaining_mines - m <= len(isolated_cells)
        )

        if total_global_configs == 0:
            raise ValueError("数学矛盾，盘面无解！")

        probabilities, info_gains, exp_remaining, global_reveal_dist = (
            {},
            defaultdict(float),
            defaultdict(float),
            defaultdict(lambda: defaultdict(int)),
        )

        if isolated_cells:
            iso_mine_configs = sum(
                w * fast_comb(len(isolated_cells) - 1, remaining_mines - m - 1)
                for m, w in all_blocks_dp.items()
                if 1 <= remaining_mines - m <= len(isolated_cells)
            )

            iso_prob = iso_mine_configs / total_global_configs
            for c in isolated_cells:
                probabilities[c], info_gains[c], exp_remaining[c] = iso_prob, 0.0, total_global_configs

        for i, b_sol in enumerate(block_solutions):
            other_dp = other_dps[i]
            b_cells = blocks[i]["cells"]
            for b_mines, b_data in b_sol.items():
                valid_global_ways = sum(
                    o_w * fast_comb(len(isolated_cells), remaining_mines - (b_mines + o_m))
                    for o_m, o_w in other_dp.items()
                    if 0 <= remaining_mines - (b_mines + o_m) <= len(isolated_cells)
                )

                for cell in b_cells:
                    probabilities[cell] = (
                        probabilities.get(cell, 0.0)
                        + (b_data["cell_mine_count"].get(cell, 0) * valid_global_ways) / total_global_configs
                    )
                    for v, count in b_data["cell_reveal_dist"].get(cell, {}).items():
                        global_reveal_dist[cell][v] += count * valid_global_ways

        for cell in frontier_cells:
            dist = global_reveal_dist[cell]
            total_safe = sum(dist.values())
            if total_safe > 0:
                entropy, expected_rem = 0.0, 0.0
                for v, count in dist.items():
                    p_v = count / total_safe
                    if p_v > 0:
                        expected_rem += p_v * count
                        v_entropy = -(p_v * math.log2(p_v))
                        if v == 0:
                            v_entropy *= 1.5
                        entropy += v_entropy

                info_gains[cell], exp_remaining[cell] = entropy, expected_rem

        certain_safe = [c for c, p in probabilities.items() if p < 1e-9]
        certain_mine = [c for c, p in probabilities.items() if p > 1.0 - 1e-9]
        calc_time_ms = (time.perf_counter() - t_start) * 1000

        sorted_candidates = sorted(probabilities.keys(), key=lambda c: (round(probabilities[c], 5), -info_gains[c]))

        top_candidates = [
            {
                "cell": c,
                "prob": probabilities[c],
                "info_gain": info_gains[c],
                "exp_rem": exp_remaining[c],
                "type": "边缘推导" if c in frontier_cells else "内部盲狙",
            }
            for c in sorted_candidates[:3]
        ]

        details = {
            "blocks": len(blocks),
            "universes": int(total_global_configs),
            "safe_found": len(certain_safe),
            "mine_found": len(certain_mine),
            "baseline_prob": baseline_prob,
            "top_candidates": top_candidates,
        }

        decisions = []
        for cell in certain_mine:
            decisions.append(
                {
                    "action": "FLAG",
                    "cell": cell,
                    "prob": 1.0,
                    "info_gain": 0.0,
                    "calc_time": calc_time_ms,
                    "details": details,
                    "debug": self._make_debug_info(
                        board, cell, "全局枚举-必雷", 1.0, info_gains[cell], frontier_cells, isolated_cells
                    ),
                }
            )

        for cell in certain_safe:
            decisions.append(
                {
                    "action": "CLICK",
                    "cell": cell,
                    "prob": 0.0,
                    "info_gain": 0.0,
                    "calc_time": calc_time_ms,
                    "details": details,
                    "debug": self._make_debug_info(
                        board, cell, "全局枚举-必安全", 0.0, info_gains[cell], frontier_cells, isolated_cells
                    ),
                }
            )

        if decisions:
            for idx, d in enumerate(decisions):
                d["batch_info"] = (idx + 1, len(decisions))
            return decisions

        best_guess = sorted_candidates[0]
        guess_type = "边缘决策" if best_guess in frontier_cells else "内部盲狙"
        return [
            {
                "action": f"GUESS ({guess_type})",
                "cell": best_guess,
                "prob": probabilities[best_guess],
                "info_gain": info_gains[best_guess],
                "calc_time": calc_time_ms,
                "details": details,
                "debug": self._make_debug_info(
                    board,
                    best_guess,
                    f"概率猜测-{guess_type}",
                    probabilities[best_guess],
                    info_gains[best_guess],
                    frontier_cells,
                    isolated_cells,
                ),
                "batch_info": (1, 1),
            }
        ]

    def _compute_block_solution(self, block):
        """计算一个独立块的全局配置统计，返回 {雷数: {cell_mine_count, cell_reveal_dist}}"""

        b_cells, b_eqs = block["cells"], block["equations"]
        num_cells, num_eqs = len(b_cells), len(b_eqs)
        cell_eq_count = dict.fromkeys(b_cells, 0)
        for eq in b_eqs:
            for c in eq["cells"]:
                cell_eq_count[c] += 1
        b_cells.sort(key=lambda c: -cell_eq_count[c])

        cell_to_idx = {c: i for i, c in enumerate(b_cells)}
        cell_to_eqs = np.full((num_cells, 8), -1, dtype=np.int32)
        cell_eq_counts = np.zeros(num_cells, dtype=np.int32)
        eq_targets = np.zeros(num_eqs, dtype=np.int8)
        eq_unassigned = np.zeros(num_eqs, dtype=np.int8)

        for i, eq in enumerate(b_eqs):
            eq_targets[i], eq_unassigned[i] = eq["target"], len(eq["cells"])
            for c in eq["cells"]:
                c_idx = cell_to_idx[c]
                cell_to_eqs[c_idx, cell_eq_counts[c_idx]] = i
                cell_eq_counts[c_idx] += 1

        cell_neighbors_matrix = np.zeros((num_cells, num_cells), dtype=np.int8)
        for i, c in enumerate(b_cells):
            for n in self.get_neighbors(*c):
                if n in cell_to_idx:
                    cell_neighbors_matrix[i, cell_to_idx[n]] = 1

        max_mines = num_cells + 1
        config_counts = np.zeros(max_mines, dtype=np.int64)
        cell_mine_counts = np.zeros((max_mines, num_cells), dtype=np.int64)
        cell_reveal_dists = np.zeros((max_mines, num_cells, 9), dtype=np.int64)

        dfs_numba(
            0,
            num_cells,
            cell_to_eqs,
            cell_eq_counts,
            eq_targets,
            cell_neighbors_matrix,
            eq_unassigned,
            np.zeros(num_eqs, dtype=np.int8),
            np.full(num_cells, -1, dtype=np.int8),
            np.zeros(num_cells, dtype=np.int8),
            config_counts,
            cell_mine_counts,
            cell_reveal_dists,
        )

        res_dict = {}
        for mines in range(max_mines):
            if config_counts[mines] > 0:
                data: dict[str, Any] = {
                    "config_count": int(config_counts[mines]),
                    "cell_mine_count": {},
                    "cell_reveal_dist": {},
                }
                for i, c in enumerate(b_cells):
                    if cell_mine_counts[mines, i] > 0:
                        data["cell_mine_count"][c] = int(cell_mine_counts[mines, i])
                    dist_dict = {
                        r_val: int(cell_reveal_dists[mines, i, r_val])
                        for r_val in range(9)
                        if cell_reveal_dists[mines, i, r_val] > 0
                    }
                    if dist_dict:
                        data["cell_reveal_dist"][c] = dist_dict
                res_dict[mines] = data

        return res_dict

    def _make_debug_info(self, board, cell, source, mine_prob, info_gain, frontier_cells=None, isolated_cells=None):
        """构建一个包含详细线索和区域分类的调试信息字典"""

        frontier_cells, isolated_cells = frontier_cells or set(), isolated_cells or set()
        clues = self._collect_supporting_clues(board, cell)
        if cell in frontier_cells:
            region = "边界格"
        elif cell in isolated_cells:
            region = "孤立格"
        else:
            region = "局部规则格"

        return {
            "source": source,
            "mine_prob": float(mine_prob),
            "safe_prob": float(1.0 - mine_prob),
            "info_gain": float(info_gain),
            "region": region,
            "support_clues": len(clues),
            "clues": clues,
        }
