import math
import sys
import time
from collections import defaultdict
from functools import lru_cache

import numpy as np
from numba import njit


@lru_cache(maxsize=None)
def fast_comb(n, k):
    if k < 0 or k > n:
        return 0
    return math.comb(n, k)


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
    if c_idx == num_cells:
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
        return

    for val in range(2):
        current_assignment[c_idx] = val
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

        if is_valid:
            if val == 1:
                for j in range(num_cells):
                    if cell_neighbors_matrix[c_idx, j] == 1:
                        current_reveal_mines[j] += 1

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

            if val == 1:
                for j in range(num_cells):
                    if cell_neighbors_matrix[c_idx, j] == 1:
                        current_reveal_mines[j] -= 1

        for k in range(cell_eq_counts[c_idx]):
            eq_idx = cell_to_eqs[c_idx, k]
            eq_unassigned[eq_idx] += 1
            if val == 1:
                eq_current_mines[eq_idx] -= 1

    current_assignment[c_idx] = -1


class ExpertMinesweeperSolver:
    def __init__(self, rows=16, cols=30, total_mines=99):
        self.rows = rows
        self.cols = cols
        self.total_mines = total_mines
        self._is_jitted = False
        # 每次 solve_step 调用后填充：疑似误识别格子坐标及其识别值
        # 格式: list of (row, col, detected_value)
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
        return self.neighbor_map[(r, c)]

    def _collect_supporting_clues(self, board, cell):
        clues = []
        for nr, nc in self.get_neighbors(*cell):
            val = board[nr][nc]
            if not (isinstance(val, int) and val > 0):
                continue

            unknowns = []
            local_flags = 0
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

    def _make_debug_info(self, board, cell, source, mine_prob, info_gain, frontier_cells=None, isolated_cells=None):
        frontier_cells = frontier_cells or set()
        isolated_cells = isolated_cells or set()
        clue_details = self._collect_supporting_clues(board, cell)

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
            "support_clues": len(clue_details),
            "clues": clue_details,
        }

    def solve_step(self, board):
        if not self._is_jitted:
            dummy_c2e = np.zeros((1, 8), dtype=np.int32)
            dummy_cec = np.zeros(1, dtype=np.int32)
            dummy_targets = np.array([0], dtype=np.int8)
            dummy_neighbors = np.array([[0]], dtype=np.int8)
            dummy_un = np.array([1], dtype=np.int8)
            dummy_cur = np.array([0], dtype=np.int8)
            dummy_assign = np.array([-1], dtype=np.int8)
            dummy_rev = np.zeros(1, dtype=np.int8)
            cfg = np.zeros(2, dtype=np.int64)
            cmc = np.zeros((2, 1), dtype=np.int64)
            crd = np.zeros((2, 1, 9), dtype=np.int64)
            dfs_numba(
                0,
                1,
                dummy_c2e,
                dummy_cec,
                dummy_targets,
                dummy_neighbors,
                dummy_un,
                dummy_cur,
                dummy_assign,
                dummy_rev,
                cfg,
                cmc,
                crd,
            )
            self._is_jitted = True

        t_start = time.perf_counter()
        self.suspicious_cells = []

        equations = []
        frontier_cells = set()
        flag_count = 0
        unknown_cells = set()
        trivial_safe = set()
        trivial_mine = set()

        for r in range(self.rows):
            for c in range(self.cols):
                val = board[r][c]
                if val == "F":
                    flag_count += 1
                elif val == -1:
                    unknown_cells.add((r, c))
                elif isinstance(val, int) and val > 0:
                    unknowns = []
                    local_flags = 0
                    for nr, nc in self.get_neighbors(r, c):
                        if board[nr][nc] == -1:
                            unknowns.append((nr, nc))
                        elif board[nr][nc] == "F":
                            local_flags += 1

                    # 标记数超过数值 → 误识别，无论是否有未知邻居都跳过
                    if local_flags > val:
                        print(
                            f"⚠️  [识别警告] ({r},{c}) 数字={val} 但周围旗帜已达 {local_flags} 个，疑似误识别，已跳过",
                            file=sys.stderr,
                        )
                        self.suspicious_cells.append((r, c, val))
                        continue

                    if unknowns:
                        target = val - local_flags
                        if target > len(unknowns):
                            # CNN 误识别：数字超出可用未知格数，忽略该约束并记录可疑格子
                            print(
                                f"⚠️  [识别警告] ({r},{c}) 数字={val} 但可用未知邻居仅 {len(unknowns)} 个，疑似误识别，已跳过",
                                file=sys.stderr,
                            )
                            self.suspicious_cells.append((r, c, val))
                            continue

                        if target == 0:
                            trivial_safe.update(unknowns)
                        elif target == len(unknowns):
                            trivial_mine.update(unknowns)
                        else:
                            frontier_cells.update(unknowns)
                            equations.append({"cells": set(unknowns), "target": target})

        remaining_mines = self.total_mines - flag_count
        if remaining_mines < 0:
            raise ValueError("标记雷数超出上限！")

        total_unknowns = len(unknown_cells)
        baseline_prob = remaining_mines / max(1, total_unknowns)

        if trivial_safe.intersection(trivial_mine):
            raise ValueError("平凡推导产生矛盾！")

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

            total_d = len(decisions)
            for idx, d in enumerate(decisions):
                d["batch_info"] = (idx + 1, total_d)
            return decisions

        isolated_cells = unknown_cells - frontier_cells

        cell_to_eq_indices = defaultdict(list)
        for i, eq in enumerate(equations):
            for cell in eq["cells"]:
                cell_to_eq_indices[cell].append(i)

        blocks = []
        eq_used = [False] * len(equations)
        for i in range(len(equations)):
            if eq_used[i]:
                continue
            current_block_eqs = []
            current_block_cells = set()
            queue = [i]
            eq_used[i] = True

            while queue:
                curr_idx = queue.pop(0)
                curr_eq = equations[curr_idx]
                current_block_eqs.append(curr_eq)
                for c in curr_eq["cells"]:
                    if c not in current_block_cells:
                        current_block_cells.add(c)
                        for neighbor_eq_idx in cell_to_eq_indices[c]:
                            if not eq_used[neighbor_eq_idx]:
                                eq_used[neighbor_eq_idx] = True
                                queue.append(neighbor_eq_idx)

            blocks.append({"cells": list(current_block_cells), "equations": current_block_eqs})

        block_solutions = []
        for block in blocks:
            b_cells = block["cells"]
            b_eqs = block["equations"]
            num_cells = len(b_cells)
            num_eqs = len(b_eqs)

            cell_eq_count = {c: 0 for c in b_cells}
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
                eq_targets[i] = eq["target"]
                eq_unassigned[i] = len(eq["cells"])
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
            eq_current_mines = np.zeros(num_eqs, dtype=np.int8)
            current_assignment = np.full(num_cells, -1, dtype=np.int8)
            current_reveal_mines = np.zeros(num_cells, dtype=np.int8)

            dfs_numba(
                0,
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

            res_dict = {}
            for mines in range(max_mines):
                if config_counts[mines] > 0:
                    data = {
                        "config_count": int(config_counts[mines]),
                        "cell_mine_count": {},
                        "cell_reveal_dist": {},
                    }
                    for i, c in enumerate(b_cells):
                        if cell_mine_counts[mines, i] > 0:
                            data["cell_mine_count"][c] = int(cell_mine_counts[mines, i])
                        dist_dict = {}
                        for r_val in range(9):
                            if cell_reveal_dists[mines, i, r_val] > 0:
                                dist_dict[r_val] = int(cell_reveal_dists[mines, i, r_val])
                        if dist_dict:
                            data["cell_reveal_dist"][c] = dist_dict
                    res_dict[mines] = data
            block_solutions.append(res_dict)

        def get_dp_combinations(blocks_to_use):
            dp = {0: 1}
            for b_sol in blocks_to_use:
                new_dp = defaultdict(int)
                for prev_mines, prev_ways in dp.items():
                    for b_mines, b_data in b_sol.items():
                        total_m = prev_mines + b_mines
                        if total_m <= remaining_mines:
                            new_dp[total_m] += prev_ways * b_data["config_count"]
                dp = new_dp
            return dp

        all_blocks_dp = get_dp_combinations(block_solutions)
        total_global_configs = sum(
            ways * fast_comb(len(isolated_cells), remaining_mines - m_blocks)
            for m_blocks, ways in all_blocks_dp.items()
            if 0 <= remaining_mines - m_blocks <= len(isolated_cells)
        )

        if total_global_configs == 0:
            raise ValueError("数学矛盾，盘面无解！")

        probabilities = {}
        info_gains = defaultdict(float)
        exp_remaining = defaultdict(float)
        global_reveal_dist = defaultdict(lambda: defaultdict(int))

        if isolated_cells:
            iso_mine_configs = sum(
                ways * fast_comb(len(isolated_cells) - 1, remaining_mines - m_blocks - 1)
                for m_blocks, ways in all_blocks_dp.items()
                if 1 <= remaining_mines - m_blocks <= len(isolated_cells)
            )
            iso_prob = iso_mine_configs / total_global_configs
            for c in isolated_cells:
                probabilities[c] = iso_prob
                info_gains[c] = 0.0
                exp_remaining[c] = total_global_configs

        for i, b_sol in enumerate(block_solutions):
            other_blocks = block_solutions[:i] + block_solutions[i + 1 :]
            other_dp = get_dp_combinations(other_blocks)
            b_cells = blocks[i]["cells"]

            for b_mines, b_data in b_sol.items():
                valid_global_ways = sum(
                    o_ways * fast_comb(len(isolated_cells), remaining_mines - (b_mines + o_mines))
                    for o_mines, o_ways in other_dp.items()
                    if 0 <= remaining_mines - (b_mines + o_mines) <= len(isolated_cells)
                )
                for cell in b_cells:
                    prob = (b_data["cell_mine_count"].get(cell, 0) * valid_global_ways) / total_global_configs
                    probabilities[cell] = probabilities.get(cell, 0.0) + prob
                    for v, count in b_data["cell_reveal_dist"].get(cell, {}).items():
                        global_reveal_dist[cell][v] += count * valid_global_ways

        for cell in frontier_cells:
            dist = global_reveal_dist[cell]
            total_safe_configs = sum(dist.values())

            if total_safe_configs > 0:
                entropy = 0.0
                expected_rem = 0.0

                for v, count in dist.items():
                    p_v = count / total_safe_configs
                    if p_v > 0:
                        expected_rem += p_v * count
                        v_entropy = -(p_v * math.log2(p_v))
                        if v == 0:
                            v_entropy *= 1.5

                        entropy += v_entropy

                info_gains[cell] = entropy
                exp_remaining[cell] = expected_rem
            else:
                info_gains[cell] = 0.0
                exp_remaining[cell] = total_global_configs

        certain_safe = [c for c, p in probabilities.items() if p < 1e-9]
        certain_mine = [c for c, p in probabilities.items() if p > 1.0 - 1e-9]

        calc_time_ms = (time.perf_counter() - t_start) * 1000

        def sort_key(c):
            prob = probabilities[c]
            bucket_prob = round(prob, 5)
            return (bucket_prob, -info_gains[c])

        sorted_candidates = sorted(probabilities.keys(), key=sort_key)

        top_candidates = []
        for c in sorted_candidates[:3]:
            top_candidates.append(
                {
                    "cell": c,
                    "prob": probabilities[c],
                    "info_gain": info_gains[c],
                    "exp_rem": exp_remaining[c],
                    "type": "边缘推导" if c in frontier_cells else "内部盲狙",
                }
            )

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
            total_d = len(decisions)
            for idx, d in enumerate(decisions):
                d["batch_info"] = (idx + 1, total_d)
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
