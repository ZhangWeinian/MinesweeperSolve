import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import psutil
import pyautogui

from src.app.manager.img2num import CellRecognizer, DatasetCollector
from src.app.manager.MouseController import BotState, click, right_click, start_keyboard_listener
from src.app.manager.TerminalPrint import Colors as C
from src.app.manager.TerminalPrint import get_local_grid_str, print_board_matrix_for_debug, print_boxed_report
from src.app.manager.Screenshot import ScreenCapture
from src.app.manager.MathematicalSolver import ExpertMinesweeperSolver

pyautogui.PAUSE = 0.0

MAX_BATCH_FLAGS = 5
ACTION_DELAY_SECONDS = 0.85
_MAX_SCAN_WORKERS = min(8, os.cpu_count() or 4)


def _recognize_row(capture, recognizer, img_np, row):
    """处理单行所有格子的识别（线程安全：仅读取 numpy + cv2 运算）。"""

    results = []
    for c in range(capture.cols):
        cell_img = capture.get_cell_image(img_np, row, c)
        results.append((row, c, recognizer.identify(cell_img), cell_img))

    return results


def _scan_full_board(capture, recognizer, state, collector=None):
    """扫描完整棋盘，返回 (board, stats, cell_images)。"""
    if state.stop:
        os._exit(0)

    img_np = capture.grab_frame()
    all_results = []
    with ThreadPoolExecutor(max_workers=_MAX_SCAN_WORKERS) as pool:
        futures = [pool.submit(_recognize_row, capture, recognizer, img_np, r) for r in range(capture.rows)]
        for f in futures:
            all_results.extend(f.result())

    board = [[-1] * capture.cols for _ in range(capture.rows)]
    stats = {"blind": 0, "blank": 0, "flag": 0, "number": 0}
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


def _count_numeric_neighbors(board, cell):
    r, c = cell
    rows = len(board)
    cols = len(board[0]) if rows else 0
    support = 0

    for nr in range(max(0, r - 1), min(rows, r + 2)):
        for nc in range(max(0, c - 1), min(cols, c + 2)):
            if (nr, nc) == (r, c):
                continue
            val = board[nr][nc]
            if isinstance(val, int) and val > 0:
                support += 1

    return support


def _select_execution_batch(board, decisions):
    flags = [d for d in decisions if d["action"] == "FLAG"]
    clicks = [d for d in decisions if d["action"] == "CLICK"]

    def rank_key(decision):
        support = _count_numeric_neighbors(board, decision["cell"])
        return (-support, decision["cell"][0], decision["cell"][1])

    flags.sort(key=rank_key)
    clicks.sort(key=rank_key)

    if flags:
        return flags[:MAX_BATCH_FLAGS]
    elif clicks:
        return clicks[:1]
    else:
        return decisions[:1]


def _format_clue_line(clue):
    rr, cc = clue["cell"]
    return (
        f"      └ ({rr:02d},{cc:02d})={clue['value']} | "
        f"周旗 {clue['flags']} | 周盲 {clue['unknowns']} | 剩余雷 {clue['target']}"
    )


def _build_debug_section(board, decision):
    debug = decision.get("debug", {})
    mine_prob = debug.get("mine_prob", decision.get("prob", 0.0)) * 100
    safe_prob = debug.get("safe_prob", 1.0 - decision.get("prob", 0.0)) * 100
    info_gain = debug.get("info_gain", decision.get("info_gain", 0.0))
    support = _count_numeric_neighbors(board, decision["cell"])

    lines = [
        f"{C.BOLD}▶ 调试剖析:{C.RESET}",
        f"  来源 : {C.CYAN}{debug.get('source', '未知')}{C.RESET} | 区域 : {C.CYAN}{debug.get('region', '未知')}{C.RESET}",
        f"  概率 : 安全 {C.GREEN}{safe_prob:06.2f}%{C.RESET} | 地雷 {C.RED}{mine_prob:06.2f}%{C.RESET}",
        f"  支撑 : 邻接数字 {C.YELLOW}{support}{C.RESET} 个 | 相关线索 {C.YELLOW}{debug.get('support_clues', 0)}{C.RESET} 条 | 熵增益 {C.MAGENTA}{info_gain:.3f}{C.RESET} bits",
    ]

    clues = debug.get("clues", [])
    if clues:
        lines.append(f"{C.BOLD}▶ 关联线索:{C.RESET}")
        for clue in clues[:5]:
            lines.append(_format_clue_line(clue))
        if len(clues) > 5:
            lines.append(f"      └ ... 其余 {len(clues) - 5} 条线索已省略")
    else:
        lines.append(f"{C.BOLD}▶ 关联线索:{C.RESET} 无（该格当前不受已开数字直接约束）")

    return lines


def _init_bot_components(rows, cols, total_mines, model_path, meta_path, root_path):
    """初始化所有核心组件并完成光学校准。"""

    intro_msg = [
        f"{C.CYAN}{C.BOLD}" + "=" * 60,
        "🚀 【自动扫雷123】",
        "=" * 60 + f"{C.RESET}",
    ]
    print("\n".join(intro_msg))

    capture = ScreenCapture(rows=rows, cols=cols)
    recognizer = CellRecognizer(model_path, meta_path)
    solver = ExpertMinesweeperSolver(rows=rows, cols=cols, total_mines=total_mines)
    collector = DatasetCollector(root_path)

    capture.calibrate()

    ready_msg = [
        f"\n{C.GREEN}{C.BOLD}" + "★" * 60,
        "🤖 自动驾驶已接管！",
        "💡 热键控制表:",
        "   ▶ 【ESC】键: 彻底退出程序",
        "★" * 60 + f"{C.RESET}\n",
    ]
    print("\n".join(ready_msg))

    return capture, recognizer, solver, collector


def _check_win_conditions(stats, total_mines, collector):
    """检查是否满足胜利或标满雷的结束条件。返回 True 表示应该终止循环。"""

    if stats["blind"] == 0:
        print(f"\n{C.GREEN}{C.BOLD}🎉🎉🎉 全图盲区清零，排雷成功！ 🎉🎉🎉{C.RESET}")
        collector.reset_session()
        return True
    elif stats["flag"] >= total_mines and stats["blind"] > 0:
        print(f"\n{C.MAGENTA}🏆 已标记满 {total_mines} 颗雷！{C.RESET}")
        collector.reset_session()
        return True
    else:
        return False


def _resolve_next_decision(board, execution_queue, solver, cell_images, collector, state):
    """处理执行队列或请求求解器计算下一步。返回 decision 或 None(表示遇到错误需跳过)。"""

    while execution_queue and board[execution_queue[0]["cell"]] != -1:
        execution_queue.pop(0)

    if execution_queue:
        return execution_queue.pop(0)

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
        return None

    for r_err, c_err, val_err in solver.suspicious_cells:
        img_err = cell_images.get((r_err, c_err))
        if img_err is not None:
            collector.save_error(img_err, val_err)
            print(
                f"  {C.YELLOW}[数据集] 误识别样本已存档: ({r_err},{c_err}) label={val_err}{C.RESET}",
                file=sys.stderr,
            )

    decisions = _select_execution_batch(board, decisions)
    total_d = len(decisions)
    for idx, d in enumerate(decisions):
        d["batch_info"] = (idx + 1, total_d)

    execution_queue.extend(decisions)
    return execution_queue.pop(0)


def _format_radar_section(stats, calc_time, details):
    """构建控制台输出的雷达与演算指标面板。"""

    cpu_usage = psutil.cpu_percent()
    ram_usage = psutil.virtual_memory().percent
    universe_str = f"{details['universes']:,}"

    return [
        f"{C.BOLD}▶ 盘面雷达:{C.RESET}",
        f"  盲区 {C.CYAN}{stats['blind']:03d}{C.RESET} | "
        f"空白 {C.CYAN}{stats['blank']:03d}{C.RESET} | "
        f"旗帜 {C.CYAN}{stats['flag']:03d}{C.RESET} | "
        f"数字 {C.CYAN}{stats['number']:03d}{C.RESET}",
        f"{C.BOLD}▶ 演算指标:{C.RESET}",
        f"  性能: CPU {C.YELLOW}{cpu_usage:04.1f}%{C.RESET} "
        f"RAM {C.YELLOW}{ram_usage:04.1f}%{C.RESET} | "
        f"单批脑力: {C.MAGENTA}{calc_time:.1f}ms{C.RESET}",
        f"  全局: 拓扑边界 {details['blocks']} 块 | 合法宇宙 {C.CYAN}{universe_str} 种{C.RESET}",
    ]


def _handle_deterministic_action(decision, step_count, section_radar, details, board, capture):
    """处理 CLICK 和 FLAG 的 UI 渲染与物理执行。"""
    action = decision["action"]
    r, c = decision["cell"]
    batch_idx, batch_total = decision["batch_info"]

    batch_str = f" {C.YELLOW}  ⚡[连击 {batch_idx}/{batch_total}]{C.RESET}" if batch_total > 1 else ""
    title = f"{C.BOLD}[回合 {step_count:03d}]{C.RESET}{batch_str}"

    section_action = []
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
                f"  预估 : 存活 {C.GREEN}100.00%{C.RESET} / 阵亡 {C.RED}0.00%{C.RESET}",
            ]
        )
    else:
        section_action.extend(
            [
                f"{C.BOLD}▶ 战术执行:{C.RESET}",
                f"  动作 : {C.RED}[右键布置信标]{C.RESET}",
                f"  坐标 : {C.MAGENTA}({r:02d}, {c:02d}){C.RESET}",
                f"  预估 : 命中 {C.GREEN}100.00%{C.RESET} / 踩空 {C.RED}0.00%{C.RESET}",
            ]
        )

    action_type = "GUESS" if "GUESS" in action else action
    local_grid_lines = get_local_grid_str(board, action_type, r, c, radius=2)
    section_action.extend(local_grid_lines)
    section_debug = _build_debug_section(board, decision)

    print_boxed_report(title, [section_radar, section_action, section_debug], box_color)

    target_x, target_y = capture.get_cell_center(r, c)
    if action == "CLICK":
        click(target_x, target_y)
    else:
        right_click(target_x, target_y)

    time.sleep(ACTION_DELAY_SECONDS)


def _handle_guess_action(decision, step_count, section_radar, details, board, capture, state):
    """处理 GUESS 的 UI 渲染、挂起等待与人工接管逻辑。"""

    r, c = decision["cell"]
    batch_idx, batch_total = decision["batch_info"]

    batch_str = f" {C.YELLOW}  ⚡[连击 {batch_idx}/{batch_total}]{C.RESET}" if batch_total > 1 else ""
    title = f"{C.BOLD}[回合 {step_count:03d}]{C.RESET}{batch_str}"

    box_color = C.YELLOW
    baseline_surv = (1.0 - details["baseline_prob"]) * 100

    section_action = [
        f"{C.BOLD}▶ {C.YELLOW}防线击穿，转入【香农信息熵决策树】模式:{C.RESET}",
        f"  基准存活率 : {C.CYAN}{baseline_surv:.2f}%{C.RESET} (全局盲狙黑区概率)",
        f"{C.BOLD}▶ Top 3 分析树:{C.RESET}",
    ]

    candidates = details["top_candidates"]
    for idx, cand in enumerate(candidates):
        is_best = idx == 0
        tag = f"{C.GREEN}[最佳]{C.RESET}" if is_best else f"{C.YELLOW}[备选]{C.RESET}"
        cr, cc = cand["cell"]
        surv = (1.0 - cand["prob"]) * 100
        gain = cand["info_gain"]
        exp_rem = cand["exp_rem"]
        ctype = cand["type"]

        surv_str = f"{C.GREEN if surv >= 50 else C.RED}{surv:.2f}%{C.RESET}"
        branch = "┣" if idx < len(candidates) - 1 else "┗"

        section_action.append(
            f"  {branch} {tag} {C.MAGENTA}({cr:02d}, {cc:02d}){C.RESET} {ctype} | 预估存活: {surv_str}"
        )
        if ctype != "内部盲狙":
            rec_act = f"{C.GREEN}[左键]{C.RESET}" if surv >= 50 else f"{C.RED}[右键]{C.RESET}"
            section_action.append(
                f"      └ 建议: {rec_act} | "
                f"香农熵增益: {C.CYAN}{gain:.3f} bits{C.RESET} | "
                f"剩余宇宙期望: {C.CYAN}{exp_rem:,.0f} 种{C.RESET}"
            )

    section_debug = _build_debug_section(board, decision)
    print_boxed_report(title, [section_radar, section_action, section_debug], box_color)

    target_x, target_y = capture.get_cell_center(r, c)

    wait_msg = [
        f"\n{C.MAGENTA}✨ 准星已自动锁定最佳决策点位 ({r:02d}, {c:02d})！{C.RESET}",
        f"👉 {C.YELLOW}[挂起] 请选择操作 (无需切屏即可生效)：{C.RESET}",
        f"   {C.GREEN}【← 左方向键】{C.RESET} : 采纳建议，让 AI 左键点开该格",
        f"   {C.RED}【→ 右方向键】{C.RESET} : 采纳建议，让 AI 右键标记该格",
        f"   {C.CYAN}【Enter 回车】{C.RESET} : 我已手动操作，直接让 AI 继续扫描",
    ]
    print("\n".join(wait_msg))

    state.waiting = True
    state.decision = None
    while state.waiting and state.decision is None and not state.stop:
        time.sleep(0.05)
    state.waiting = False

    if state.decision == "left":
        print(f"🤖 {C.GREEN}接收指令：自动代劳 [左键点击]...{C.RESET}")
        click(target_x, target_y)
        time.sleep(0.2)
    elif state.decision == "right":
        print(f"🤖 {C.RED}接收指令：自动代劳 [右键标雷]...{C.RESET}")
        right_click(target_x, target_y)
        time.sleep(0.2)
    elif state.decision == "enter":
        print(f"🤖 {C.CYAN}接收指令：您已手动操作，继续推进...{C.RESET}")
        time.sleep(0.2)


def run_auto_bot(
    rows: int = 16, cols: int = 30, total_mines: int = 99, model_path=None, meta_path=None, root_path=None
):
    state = BotState()
    capture, recognizer, solver, collector = _init_bot_components(
        rows, cols, total_mines, model_path, meta_path, root_path
    )

    start_keyboard_listener(state)

    step_count = 0
    execution_queue = []

    while not state.stop:
        board_data = _scan_full_board(capture, recognizer, state, collector)

        if state.stop:
            os._exit(0)

        if board_data[0] is None:
            continue

        board, stats, cell_images = board_data

        if _check_win_conditions(stats, total_mines, collector):
            break

        decision = _resolve_next_decision(board, execution_queue, solver, cell_images, collector, state)

        if decision is None:
            continue

        step_count += 1
        action = decision["action"]
        r, c = decision["cell"]
        calc_time = decision["calc_time"]
        details = decision["details"]

        center = capture.get_cell_center(r, c)
        if center is None:
            print(f"{C.RED}❌ 坐标 ({r}, {c}) 未被正确初始化，跳过本回合...{C.RESET}")
            continue

        section_radar = _format_radar_section(stats, calc_time, details)

        if action in ("CLICK", "FLAG"):
            _handle_deterministic_action(decision, step_count, section_radar, details, board, capture)
        else:
            _handle_guess_action(decision, step_count, section_radar, details, board, capture, state)
