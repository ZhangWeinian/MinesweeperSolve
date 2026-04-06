import os
import re
import shutil
import unicodedata

if os.name == "nt":
    import ctypes

    kernel32 = ctypes.windll.kernel32
    handle = kernel32.GetStdHandle(-11)
    mode = ctypes.c_uint32()
    kernel32.GetConsoleMode(handle, ctypes.byref(mode))
    kernel32.SetConsoleMode(handle, mode.value | 0x0004)


class Colors:
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    MAGENTA = "\033[95m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


C = Colors


def get_visual_length(s):
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    clean_s = ansi_escape.sub("", s)
    length = 0
    for char in clean_s:
        if unicodedata.east_asian_width(char) in ("F", "W"):
            length += 2
        else:
            length += 1
    return length


def print_boxed_report(title, sections, box_color=Colors.CYAN):
    term_width = shutil.get_terminal_size((120, 20)).columns
    box_width = term_width - 1

    top_border = f"{box_color}╭" + "─" * (box_width - 2) + f"╮{Colors.RESET}"
    bottom_border = f"{box_color}╰" + "─" * (box_width - 2) + f"╯{Colors.RESET}"
    separator = f"{box_color}├" + "─" * (box_width - 2) + f"┤{Colors.RESET}"

    output_lines = [f"\n{top_border}"]

    title_vlen = get_visual_length(title)
    pad_left = max(0, (box_width - 2 - title_vlen) // 2)
    pad_right = max(0, box_width - 2 - title_vlen - pad_left)
    output_lines.append(
        f"{box_color}│{Colors.RESET}{' ' * pad_left}{title}{' ' * pad_right}{box_color}│{Colors.RESET}"
    )

    for i, sec in enumerate(sections):
        output_lines.append(separator)
        for line in sec:
            v_len = get_visual_length(line)
            pad = max(0, box_width - 4 - v_len)
            output_lines.append(
                f"{box_color}│{Colors.RESET} {line}{' ' * pad} {box_color}│{Colors.RESET}"
            )

    output_lines.append(bottom_border)
    print("\n".join(output_lines))


def print_naming_guide():
    guide = [
        f"\n{C.YELLOW}" + "=" * 50,
        f"📝 【手动标注命名规范指南】",
        "-" * 50,
        f"▶ [红色的旗帜]  -> 命名为 F.png",
        f"▶ [意外的问号]  -> 命名为 Q.png",
        f"▶ [数字 1 到 8] -> 命名为 1.png, 2.png...",
        "=" * 50 + f"{C.RESET}\n",
    ]
    print("\n".join(guide))


def print_board_matrix_for_debug(board):
    output_lines = [
        f"\n{C.RED}========================================",
        "🔍 【算法崩溃断点数据提取】",
        "========================================",
        "请将以下矩阵代码【完整复制】发给我：",
        "bug_board = [",
    ]
    for row in board:
        formatted_row = []
        for cell in row:
            if cell == -1:
                formatted_row.append(" -1")
            elif cell == "F":
                formatted_row.append("'F'")
            elif cell == "Q":
                formatted_row.append("'Q'")
            else:
                formatted_row.append(f" {cell} ")
        output_lines.append("    [" + ", ".join(formatted_row) + "],")

    output_lines.append("]")
    output_lines.append(f"========================================{C.RESET}\n")
    print("\n".join(output_lines))
