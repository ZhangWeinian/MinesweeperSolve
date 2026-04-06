import os
import time

import cv2
import numpy as np
import psutil
import pyautogui
from minesweeper_calculator import ExpertMinesweeperSolver
from pynput import keyboard
from terminal_ui import Colors as C
from terminal_ui import (
    print_board_matrix_for_debug,
    print_boxed_report,
    print_naming_guide,
)
from vision_shape_ocr import HighResShapeOCR

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UNKNOWN_DIR = os.path.join(BASE_DIR, "templates_unknown")

pyautogui.PAUSE = 0.0
STOP_BOT = False
WAITING_FOR_DECISION = False
USER_DECISION = None
error_counts_dict = {}


def scan_full_board(vision):
    global error_counts_dict, WAITING_FOR_DECISION, USER_DECISION
    if STOP_BOT:
        os._exit(0)

    sct_img = vision.sct.grab(vision.monitor)
    img_np = np.array(sct_img)[:, :, :3]

    board = [[-1 for _ in range(vision.cols)] for _ in range(vision.rows)]
    stats = {"blind": 0, "blank": 0, "flag": 0, "number": 0}

    for r in range(vision.rows):
        if STOP_BOT:
            os._exit(0)
        for c in range(vision.cols):
            y1, y2, x1, x2 = vision.grid_map[r][c]["slice"]
            cell_img = img_np[y1:y2, x1:x2]

            coord_key = f"{r}_{c}"
            error_times = error_counts_dict.get(coord_key, 0)

            force_guess = error_times >= 3
            result = vision.identify_cell(cell_img, force_guess=force_guess)

            if result == "Q":
                target_x = vision.grid_map[r][c]["cx"]
                target_y = vision.grid_map[r][c]["cy"]
                print(
                    f"\n{C.YELLOW}🔄 侦测到误触的【？】标记于 ({r}, {c})，停顿 0.8s 后执行右键修复...{C.RESET}"
                )
                time.sleep(0.8)
                pyautogui.rightClick(target_x, target_y)
                time.sleep(0.2)
                return None, stats

            if result == "?":
                error_counts_dict[coord_key] = error_times + 1
                if not os.path.exists(UNKNOWN_DIR):
                    os.makedirs(UNKNOWN_DIR)

                for old_file in os.listdir(UNKNOWN_DIR):
                    os.remove(os.path.join(UNKNOWN_DIR, old_file))

                cv2.imwrite(
                    os.path.join(UNKNOWN_DIR, f"unknown_r{r}_c{c}_color.png"), cell_img
                )
                shape, _ = vision.binarize_cell(cell_img)
                if shape is not None:
                    cv2.imwrite(
                        os.path.join(UNKNOWN_DIR, f"unknown_r{r}_c{c}_bw.png"), shape
                    )

                err_msg = [
                    f"\n{C.RED}{C.BOLD}" + "!" * 50,
                    f"⚠️ [视觉告警] 坐标 ({r}, {c}) 出现未知字形！(历史报错: {error_times+1}/3 次)",
                    "!" * 50 + f"{C.RESET}",
                ]
                print("\n".join(err_msg))
                print_naming_guide()

                print(f"{C.MAGENTA}⏸️ 已暂停！{C.RESET}")
                print(
                    f"👉 {C.YELLOW}若为新字形，放入模板后按键盘【回车】重载并继续。{C.RESET}"
                )

                WAITING_FOR_DECISION = True
                USER_DECISION = None
                while (
                    WAITING_FOR_DECISION and USER_DECISION != "enter" and not STOP_BOT
                ):
                    time.sleep(0.05)
                WAITING_FOR_DECISION = False

                vision.load_templates()
                return None, stats

            if coord_key in error_counts_dict:
                del error_counts_dict[coord_key]

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


def run_auto_bot():
    global STOP_BOT, WAITING_FOR_DECISION, USER_DECISION

    intro_msg = [
        f"{C.CYAN}{C.BOLD}" + "=" * 60,
        "🚀 【自动扫雷123】",
        "=" * 60 + f"{C.RESET}",
    ]
    print("\n".join(intro_msg))

    vision = HighResShapeOCR(rows=16, cols=30)
    solver = ExpertMinesweeperSolver(rows=16, cols=30, total_mines=99)

    if not vision.templates:
        print(
            f"{C.YELLOW}💡 [学习模式激活] 当前视觉字库为空。AI 将在实战中收集新字形并请您手动标注！{C.RESET}"
        )

    vision.calibrate()

    ready_msg = [
        f"\n{C.GREEN}{C.BOLD}" + "★" * 60,
        "🤖 自动驾驶已接管！",
        "💡 热键控制表:",
        "   ▶ 【ESC】键: 彻底退出程序",
        "★" * 60 + f"{C.RESET}\n",
    ]
    print("\n".join(ready_msg))

    def on_key_press(key):
        global STOP_BOT, WAITING_FOR_DECISION, USER_DECISION
        if key == keyboard.Key.esc:
            print(f"\n{C.RED}[系统指令] 接收到 ESC 键，彻底终止程序...{C.RESET}")
            STOP_BOT = True
            os._exit(0)

        if WAITING_FOR_DECISION:
            if key == keyboard.Key.left:
                USER_DECISION = "left"
            elif key == keyboard.Key.right:
                USER_DECISION = "right"
            elif key == keyboard.Key.enter:
                USER_DECISION = "enter"

    listener = keyboard.Listener(on_press=on_key_press)
    listener.start()

    step_count = 0
    execution_queue = []

    while not STOP_BOT:
        board_data = scan_full_board(vision)

        if STOP_BOT:
            os._exit(0)

        if board_data[0] is None:
            continue

        board, stats = board_data

        if stats["blind"] == 0:
            print(
                f"\n{C.GREEN}{C.BOLD}����🎉🎉 全图盲区清零！排雷成功！伟大的胜利！ 🎉🎉🎉{C.RESET}"
            )
            break

        if stats["flag"] >= 99 and stats["blind"] > 0:
            print(
                f"\n{C.MAGENTA}🏆 已标记满 99 颗雷！进入【最终清场模式】，直接点开剩余 {stats['blind']} 个盲区！{C.RESET}"
            )
            execution_queue.clear()
            for r in range(vision.rows):
                for c in range(vision.cols):
                    if board[r][c] == -1:
                        cell_data = vision.grid_map[r][c]
                        if cell_data is not None:
                            pyautogui.click(cell_data["cx"], cell_data["cy"])
                            time.sleep(0.03)

            print(f"\n{C.GREEN}{C.BOLD}🎉🎉🎉 战术清场完毕，排雷成功！ 🎉🎉🎉{C.RESET}")
            break

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

                WAITING_FOR_DECISION = True
                USER_DECISION = None
                while (
                    WAITING_FOR_DECISION and USER_DECISION != "enter" and not STOP_BOT
                ):
                    time.sleep(0.05)
                WAITING_FOR_DECISION = False

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

        cell_data = vision.grid_map[r][c]
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
            f"  盲区 {C.CYAN}{stats['blind']:03d}{C.RESET} | 空白 {C.CYAN}{stats['blank']:03d}{C.RESET} | 旗帜 {C.CYAN}{stats['flag']:03d}{C.RESET} | 数字 {C.CYAN}{stats['number']:03d}{C.RESET}",
            f"{C.BOLD}▶ 演算指标:{C.RESET}",
            f"  性能: CPU {C.YELLOW}{cpu_usage:04.1f}%{C.RESET} RAM {C.YELLOW}{ram_usage:04.1f}%{C.RESET} | 单批脑力: {C.MAGENTA}{calc_time:.1f}ms{C.RESET}",
            f"  全局: 拓扑边界 {details['blocks']} 块 | 合法宇宙 {C.CYAN}{universe_str} 种{C.RESET}",
        ]

        section_action = []
        box_color = C.CYAN

        if action == "CLICK" or action == "FLAG":
            box_color = C.GREEN if action == "CLICK" else C.RED
            section_action.append(
                f"{C.BOLD}▶ 逻辑推导:{C.RESET} 绝对安全区 {C.GREEN}{details['safe_found']}{C.RESET} 个 | 绝对雷区 {C.RED}{details['mine_found']}{C.RESET} 个"
            )

            if action == "CLICK":
                section_action.append(f"{C.BOLD}▶ 战术执行:{C.RESET}")
                section_action.append(f"  动作 : {C.GREEN}[左键探索安全]{C.RESET}")
                section_action.append(
                    f"  坐标 : {C.MAGENTA}({r:02d}, {c:02d}){C.RESET}"
                )
                section_action.append(
                    f"  预估 : 存活 {C.GREEN}100.00%{C.RESET} / 阵亡 {C.RED}0.00%{C.RESET}"
                )
            else:
                section_action.append(f"{C.BOLD}▶ 战术执行:{C.RESET}")
                section_action.append(f"  动作 : {C.RED}[右键布置信标]{C.RESET}")
                section_action.append(
                    f"  坐标 : {C.MAGENTA}({r:02d}, {c:02d}){C.RESET}"
                )
                section_action.append(
                    f"  预估 : 命中 {C.GREEN}100.00%{C.RESET} / 踩空 {C.RED}0.00%{C.RESET}"
                )

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
                f"{C.BOLD}▶ {C.YELLOW}防线击穿！转入【香农信息熵决策树】模式:{C.RESET}"
            )
            section_action.append(
                f"  基准存活率 : {C.CYAN}{baseline_surv:.2f}%{C.RESET} (全局盲狙黑区概率)"
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

                rec_act = (
                    f"{C.GREEN}[左键]{C.RESET}"
                    if surv >= 50
                    else f"{C.RED}[右键]{C.RESET}"
                )
                surv_str = f"{C.GREEN if surv>=50 else C.RED}{surv:.2f}%{C.RESET}"
                branch = "┣" if idx < len(candidates) - 1 else "┗"

                section_action.append(
                    f"  {branch} {tag} {C.MAGENTA}({cr:02d}, {cc:02d}){C.RESET} {ctype} | 预估存活: {surv_str}"
                )
                if ctype != "内部盲狙":
                    section_action.append(
                        f"      └ 建议: {rec_act} | 熵增益: {C.CYAN}{gain:.2f}{C.RESET} | 坍缩预期: {C.CYAN}{exp_rem:.1f}种{C.RESET}"
                    )

            print_boxed_report(title, [section_radar, section_action], box_color)

            best_cand = candidates[0]
            best_r, best_c = best_cand["cell"]
            best_target_x = vision.grid_map[best_r][best_c]["cx"]
            best_target_y = vision.grid_map[best_r][best_c]["cy"]

            wait_msg = [
                f"\n{C.MAGENTA}✨ 准星已自动锁定最佳决策点位 ({best_r:02d}, {best_c:02d})！{C.RESET}",
                f"👉 {C.YELLOW}[挂起] 请选择操作 (无需切屏即可生效)：{C.RESET}",
                f"   {C.GREEN}【← 左方向键】{C.RESET} : 采纳建议，让 AI 左键点开该格",
                f"   {C.RED}【→ 右方向键】{C.RESET} : 采纳建议，让 AI 右键标记该格",
                f"   {C.CYAN}【Enter 回车】{C.RESET} : 我已手动操作，直接让 AI 继续扫描",
            ]
            print("\n".join(wait_msg))

            WAITING_FOR_DECISION = True
            USER_DECISION = None
            while WAITING_FOR_DECISION and USER_DECISION is None and not STOP_BOT:
                time.sleep(0.05)
            WAITING_FOR_DECISION = False

            if USER_DECISION == "left":
                print(f"🤖 {C.GREEN}接收指令：自动代劳 [左键点击]...{C.RESET}")
                pyautogui.click(best_target_x, best_target_y)
                time.sleep(0.2)
            elif USER_DECISION == "right":
                print(f"🤖 {C.RED}接收指令：自动代劳 [右键标雷]...{C.RESET}")
                pyautogui.rightClick(best_target_x, best_target_y)
                time.sleep(0.2)
            elif USER_DECISION == "enter":
                print(f"🤖 {C.CYAN}接收指令：您已手动操作，继续推进...{C.RESET}")
                time.sleep(0.2)


if __name__ == "__main__":
    import ctypes

    ctypes.windll.user32.SetProcessDPIAware()

    try:
        run_auto_bot()
    except pyautogui.FailSafeException:
        print(f"\n{C.RED}🛑 触发物理紧急制动！自动驾驶中止。{C.RESET}")
        os._exit(0)
    except KeyboardInterrupt:
        print(f"\n{C.RED}🛑 接收到 Ctrl+C 中断信号！自动驾驶彻底中止。{C.RESET}")
        os._exit(0)
