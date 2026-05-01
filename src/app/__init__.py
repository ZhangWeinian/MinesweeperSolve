import argparse
from pathlib import Path


def main(args=None, root_path: Path | None = None):
    from src.app.manager import main as _run

    _root = root_path or Path(__file__).resolve().parent.parent.parent
    _default_model = str(_root / "model" / "minesweeper_cnn.pth")
    _default_meta = str(_root / "model" / "minesweeper_meta.json")

    parser = argparse.ArgumentParser(description="扫雷 AI：自动运行模式")
    parser.add_argument("--rows", type=int, default=16, help="棋盘行数（默认 16）")
    parser.add_argument("--cols", type=int, default=30, help="棋盘列数（默认 30）")
    parser.add_argument("--mines", type=int, default=99, help="地雷总数（默认 99）")
    parser.add_argument("--model-path", default=_default_model, dest="model_path", help="CNN 模型权重路径")
    parser.add_argument("--meta-path", default=_default_meta, dest="meta_path", help="CNN 元数据路径")
    parsed = parser.parse_args(args)

    _run(
        rows=parsed.rows,
        cols=parsed.cols,
        total_mines=parsed.mines,
        cnn_model_path=Path(parsed.model_path),
        cnn_meta_path=Path(parsed.meta_path),
        root_path=_root,
    )


__all__ = ["main"]
