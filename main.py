import argparse
import platform
import sys
from pathlib import Path

ROOT_PATH = Path(__file__).resolve().parent

_USING_EXCLUSIVE = {"--rows", "--cols", "--mines", "--model-path", "--meta-path"}
_TRAIN_EXCLUSIVE = {"--test"}


def _default_mode() -> str:
    return "using" if platform.system() == "Windows" else "train"


def _check_collision(mode: str, remaining: list[str]) -> None:
    arg_keys = {tok for tok in remaining if tok.startswith("--")}
    bad = arg_keys & (_TRAIN_EXCLUSIVE if mode == "using" else _USING_EXCLUSIVE)
    if bad:
        print(
            f"❌ 错误：参数 {', '.join(sorted(bad))} 与模式 --{mode} 不兼容，" "请勿交错使用两个模块的参数。",
            file=sys.stderr,
        )
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="扫雷 AI 主入口",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "使用示例:\n"
            "  uv run main.py                                         # 平台自动选择（Windows→using，Linux→train）\n"
            "  uv run main.py --using                                 # 运行扫雷自动化（默认参数）\n"
            "  uv run main.py --using --rows 9 --cols 9 --mines 10    # 自定义棋盘\n"
            "  uv run main.py --train                                 # 训练 CNN 模型\n"
            "  uv run main.py --train --test 5                        # 仅测试（5 轮）\n"
        ),
    )
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--using", action="store_true", default=False, help="运行扫雷自动化（app 模块）")
    group.add_argument("--train", action="store_true", default=False, help="训练或测试 CNN 模型（cnn 模块）")
    args, remaining = parser.parse_known_args()

    if args.using:
        mode = "using"
    elif args.train:
        mode = "train"
    else:
        mode = _default_mode()
        print(f"ℹ️ 未指定模式，当前平台（{platform.system()}）自动选择：--{mode}")

    _check_collision(mode, remaining)

    if mode == "using":
        from src.app import main as app_main

        app_main(remaining, root_path=ROOT_PATH)
    else:
        from src.cnn import main as cnn_main

        cnn_main(remaining, root_path=ROOT_PATH)


if __name__ == "__main__":
    main()
