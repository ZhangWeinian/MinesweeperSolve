from src.cnn.minesweeper_ocr.dataset import MinesweeperDataset
from src.cnn.minesweeper_ocr.predictor import predict_image
from src.cnn.minesweeper_ocr.trainer import run_training


def main(args=None, root_path=None):
    import argparse
    import os
    from pathlib import Path

    import torch

    from src.cnn.test import run_test
    from src.export import MinesweeperCNN

    _root = root_path or Path(__file__).resolve().parent.parent.parent
    _model_path = _root / "model" / "minesweeper_cnn.pth"

    parser = argparse.ArgumentParser(description="扫雷 AI：训练与测试一体入口")
    parser.add_argument(
        "--test",
        nargs="?",
        const=None,
        help="只执行测试，不训练。可加数字 N (1~100) 进行 N 次深度测试。省略 N 默认测试 1 次。",
    )
    parsed = parser.parse_args(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    is_test_only_mode = parsed.test is not None
    test_count: int = 1

    if is_test_only_mode:
        if not _model_path.exists():
            print("❌ 错误：找不到模型文件。请先无参运行 `uv run main.py --train` 进行训练。")
            return

        model = MinesweeperCNN(num_classes=9).to(device)
        model.load_state_dict(torch.load(str(_model_path), weights_only=True))
        test_count = 1
        if parsed.test is not None:
            try:
                n = int(parsed.test)
                if 1 <= n <= 100:
                    test_count = n
                else:
                    print("⚠️ 0 < N <= 100，已重置为 1 次。")
            except ValueError:
                print("⚠️ 参数必须为整数，已重置为 1 次。")

        print("\n🧠 进入纯测试模式...\n")
        run_test(test_count, model, device, root_path=_root)

    else:
        print("🚀 进入训练模式...\n")
        trained_model_path, _ = run_training(device=device, root_path=_root)

        model = MinesweeperCNN(num_classes=9).to(device)
        model.load_state_dict(torch.load(trained_model_path, weights_only=True))

        print("\n🧠 训练完毕，开始验证测试...\n")
        run_test(1, model, device, root_path=_root)


__all__ = ["MinesweeperDataset", "run_training", "predict_image", "main"]
