"""Bot 主循环：扫描 → 求解 → 执行 → 交互。"""

import os
import time

import psutil
import pyautogui
from pynput import keyboard

from .config import COLS, ROWS, TOTAL_MINES
from .scanner import scan_full_board
from .solver import ExpertMinesweeperSolver
from .ui import Colors as C
from .ui import get_local_grid_str, print_board_matrix_for_debug, print_boxed_report
from .vision import CellRecognizer, ScreenCapture

pyautogui.PAUSE = 0.0


class BotState:
    """线程间共享的轻量状态对象（替代全局变量）。"""

    __slots__ = ("stop", "waiting", "decision")

    def __init__(self):
        self.stop = False
        self.waiting = False
        self.decision = None


def run_auto_bot():
    state = BotState()
    error_counts: dict[str, int] = {}

    intro_msg = [
        f"{C.CYAN}{C.BOLD}" + "=" * 60,
        "🚀 【自动扫雷123】",
        "=" * 60 + f"{C.RESET}",
    ]
    print("\n".join(intro_msg))

    capture = ScreenCapture(rows=ROWS, cols=COLS)
    recognizer = CellRecognizer()
    solver = ExpertMinesweeperSolver(rows=ROWS, cols=COLS, total_mines=TOTAL_MINES)

    if not recognizer.templates:
        print(
            f"{C.YELLOW}💡 [学习模式激活] 当前视觉字库为空。"
            f"AI 将在实战中收集新字形并请手动标注！{C.RESET}"
        )

    capture.calibrate()

    ready_msg = [
        f"\n{C.GREEN}{C.BOLD}" + "★" * 60,
        "🤖 自动驾驶已接管！",
        "💡 热键控制表:",
        "   ▶ 【ESC】键: 彻底退出程序",
        "★" * 60 + f"{C.RESET}\n",
    ]
    print("\n".join(ready_msg))

    def on_key_press(key):
        if key == keyboard.Key.esc:
            print(f"\n{C.RED}[系统指令] 接收到 ESC 键，彻底终止程序...{C.RESET}")
            state.stop = True
            os._exit(0)

        if state.waiting:
            if key == keyboard.Key.left:
                state.decision = "left"
            elif key == keyboard.Key.right:
                state.decision = "right"
            elif key == keyboard.Key.enter:
                state.decision = "enter"

    listener = keyboard.Listener(on_press=on_key_press)
    listener.start()

    step_count = 0
    execution_queue = []

    while not state.stop:
        board_data = scan_full_board(capture, recognizer, error_counts, state)

        if state.stop:
            os._exit(0)

        if board_data[0] is None:
            continue

        board, stats = board_data

        if stats["blind"] == 0:
            print(
                f"\n{C.GREEN}{C.BOLD}🎉🎉🎉 全图盲区清零！排雷成功！伟大的胜利！ 🎉🎉🎉{C.RESET}"
            )
            break

        if stats["flag"] >= TOTAL_MINES and stats["blind"] > 0:
            print(
                f"\n{C.MAGENTA}🏆 已标记满 {TOTAL_MINES} 颗雷！"
                f"进入【最终清场模式】，直接点开剩余 {stats['blind']} 个盲区！{C.RESET}"
            )
            execution_queue.clear()
            for r in range(ROWS):
                for c in range(COLS):
                    if board[r][c] == -1:
                        cell_data = capture.grid_map[r][c]
                        if cell_data is not None:
                            pyautogui.click(cell_data["cx"], cell_data["cy"])
                            time.sleep(0.03)

            print(f"\n{C.GREEN}{C.BOLD}🎉🎉🎉 战术清场完毕，排雷成功！ 🎉🎉🎉{C.RESET}")
            break

        # 消耗已完成的队列项
        while execution_queue:
            next_r, next_c = execution_queue[0]["cell"]
            if board[next_r][next_c] != -1:
                execution_queue.pop(0)
            else:
                break

        # 调度执行
        if execution_queue:
            decision = execution_queue.pop(0)
            step_count += 1
        else:
            try:
                decisions = solver.solve_step(board)
            except ValueError as e:
                print(f"\n{C.RED}💀 盘面产生逻辑矛盾: {e}{C.RESET}")
                print_board_matrix_for_debug(board)

                state.waiting = True
                state.decision = None
                while state.waiting and state.decision != "enter" and not state.stop:
                    time.sleep(0.05)
                state.waiting = False

                step_count -= 1
                continue

            execution_queue.extend(decisions)
            decision = execution_queue.pop(0)
            step_count += 1

        cpu_usage = psutil.cpu_percent()
        ram_usage = psutil.virtual_memory().percent

        action = decision["action"]
        r, c = decision["cell"]
        prob = decision["prob"]
        info_gain = decision["info_gain"]
        calc_time = decision["calc_time"]
        details = decision["details"]
        batch_idx, batch_total = decision["batch_info"]

        cell_data = capture.grid_map[r][c]
        if cell_data is None:
            print(f"{C.RED}❌ 坐标 ({r}, {c}) 未被正确初始化，跳过本回合...{C.RESET}")
            continue

        target_x = cell_data["cx"]
        target_y = cell_data["cy"]
        universe_str = f"{details['universes']:,}"

        batch_str = (
            f" {C.YELLOW}  ⚡[连击 {batch_idx}/{batch_total}]{C.RESET}"
            if batch_total > 1
            else ""
        )
        title = f"{C.BOLD}[回合 {step_count:03d}]{C.RESET}{batch_str}"

        section_radar = [
            f"{C.BOLD}▶ 盘面雷达:{C.RESET}",
            f"  盲区 {C.CYAN}{stats['blind']:03d}{C.RESET} | "
            f"空白 {C.CYAN}{stats['blank']:03d}{C.RESET} | "
            f"旗帜 {C.CYAN}{stats['flag']:03d}{C.RESET} | "
            f"数字 {C.CYAN}{stats['number']:03d}{C.RESET}",
            f"{C.BOLD}▶ 演算指标:{C.RESET}",
            f"  性能: CPU {C.YELLOW}{cpu_usage:04.1f}%{C.RESET} "
            f"RAM {C.YELLOW}{ram_usage:04.1f}%{C.RESET} | "
            f"单批脑力: {C.MAGENTA}{calc_time:.1f}ms{C.RESET}",
            f"  全局: 拓扑边界 {details['blocks']} 块 | "
            f"合法宇宙 {C.CYAN}{universe_str} 种{C.RESET}",
        ]

        section_action = []
        box_color = C.CYAN

        if action in ("CLICK", "FLAG"):
            box_color = C.GREEN if action == "CLICK" else C.RED
            section_action.append(
                f"{C.BOLD}▶ 逻辑推导:{C.RESET} "
                f"绝对安全区 {C.GREEN}{details['safe_found']}{C.RESET} 个 | "
                f"绝对雷区 {C.RED}{details['mine_found']}{C.RESET} 个"
            )

            if action == "CLICK":
                section_action.extend(
                    [
                        f"{C.BOLD}▶ 战术执行:{C.RESET}",
                        f"  动作 : {C.GREEN}[左键探索安全]{C.RESET}",
                        f"  坐标 : {C.MAGENTA}({r:02d}, {c:02d}){C.RESET}",
                        f"  预估 : 存活 {C.GREEN}100.00%{C.RESET} / "
                        f"阵亡 {C.RED}0.00%{C.RESET}",
                    ]
                )
            else:
                section_action.extend(
                    [
                        f"{C.BOLD}▶ 战术执行:{C.RESET}",
                        f"  动作 : {C.RED}[右键布置信标]{C.RESET}",
                        f"  坐标 : {C.MAGENTA}({r:02d}, {c:02d}){C.RESET}",
                        f"  预估 : 命中 {C.GREEN}100.00%{C.RESET} / "
                        f"踩空 {C.RED}0.00%{C.RESET}",
                    ]
                )

            action_type = "GUESS" if "GUESS" in action else action
            local_grid_lines = get_local_grid_str(board, action_type, r, c, radius=2)
            section_action.extend(local_grid_lines)

            print_boxed_report(title, [section_radar, section_action], box_color)

            if action == "CLICK":
                pyautogui.click(target_x, target_y)
            else:
                pyautogui.rightClick(target_x, target_y)

            time.sleep(0.15)

        else:
            box_color = C.YELLOW
            baseline_surv = (1.0 - details["baseline_prob"]) * 100

            section_action.append(
                f"{C.BOLD}▶ {C.YELLOW}防线击穿，转入【香农信息熵决策树】模式:{C.RESET}"
            )
            section_action.append(
                f"  基准存活率 : {C.CYAN}{baseline_surv:.2f}%{C.RESET} "
                f"(全局盲狙黑区概率)"
            )
            section_action.append(f"{C.BOLD}▶ Top 3 分析树:{C.RESET}")

            candidates = details["top_candidates"]
            for idx, cand in enumerate(candidates):
                is_best = idx == 0
                tag = (
                    f"{C.GREEN}[最佳]{C.RESET}"
                    if is_best
                    else f"{C.YELLOW}[备选]{C.RESET}"
                )
                cr, cc = cand["cell"]
                surv = (1.0 - cand["prob"]) * 100
                gain = cand["info_gain"]
                exp_rem = cand["exp_rem"]
                ctype = cand["type"]

                surv_str = f"{C.GREEN if surv >= 50 else C.RED}{surv:.2f}%{C.RESET}"
                branch = "┣" if idx < len(candidates) - 1 else "┗"

                section_action.append(
                    f"  {branch} {tag} {C.MAGENTA}({cr:02d}, {cc:02d}){C.RESET} "
                    f"{ctype} | 预估存活: {surv_str}"
                )
                if ctype != "内部盲狙":
                    rec_act = (
                        f"{C.GREEN}[左键]{C.RESET}"
                        if surv >= 50
                        else f"{C.RED}[右键]{C.RESET}"
                    )
                    section_action.append(
                        f"      └ 建议: {rec_act} | "
                        f"香农熵增益: {C.CYAN}{gain:.3f} bits{C.RESET} | "
                        f"剩余宇宙期望: {C.CYAN}{exp_rem:,.0f} 种{C.RESET}"
                    )

            print_boxed_report(title, [section_radar, section_action], box_color)

            best_cand = candidates[0]
            best_r, best_c = best_cand["cell"]
            best_target_x = capture.grid_map[best_r][best_c]["cx"]
            best_target_y = capture.grid_map[best_r][best_c]["cy"]

            wait_msg = [
                f"\n{C.MAGENTA}✨ 准星已自动锁定最佳决策点位 "
                f"({best_r:02d}, {best_c:02d})！{C.RESET}",
                f"👉 {C.YELLOW}[挂起] 请选择操作 (无需切屏即可生效)：{C.RESET}",
                f"   {C.GREEN}【← 左方向键】{C.RESET} : "
                f"采纳建议，让 AI 左键点开该格",
                f"   {C.RED}【→ 右方向键】{C.RESET} : " f"采纳建议，让 AI 右键标记该格",
                f"   {C.CYAN}【Enter 回车】{C.RESET} : "
                f"我已手动操作，直接让 AI 继续扫描",
            ]
            print("\n".join(wait_msg))

            state.waiting = True
            state.decision = None
            while state.waiting and state.decision is None and not state.stop:
                time.sleep(0.05)
            state.waiting = False

            if state.decision == "left":
                print(f"🤖 {C.GREEN}接收指令：自动代劳 [左键点击]...{C.RESET}")
                pyautogui.click(best_target_x, best_target_y)
                time.sleep(0.2)
            elif state.decision == "right":
                print(f"🤖 {C.RED}接收指令：自动代劳 [右键标雷]...{C.RESET}")
                pyautogui.rightClick(best_target_x, best_target_y)
                time.sleep(0.2)
            elif state.decision == "enter":
                print(f"🤖 {C.CYAN}接收指令：您已手动操作，继续推进...{C.RESET}")
                time.sleep(0.2)
