import json
import os
import random
import shutil
import time
from pathlib import Path

import torch

from src.cnn.minesweeper_ocr import predict_image
from src.export import MinesweeperCNN


def _load_meta_and_class_names(meta_path: str) -> tuple[dict[int, str], list[str]]:
    """Load metadata and return indexed and sorted class names."""
    if not os.path.exists(meta_path):
        print("❌ 错误：找不到元数据文件。")
        return {}, []

    with open(meta_path, "r", encoding="utf-8") as f:
        idx_to_class: dict[int, str] = json.load(f)
    class_to_idx: dict[str, int] = {v: k for k, v in idx_to_class.items()}
    class_names = sorted(class_to_idx.keys())
    return idx_to_class, class_names


def _collect_train_images(
    train_dir: str, class_names: list[str]
) -> dict[str, list[str]]:
    """Collect all training images by class."""
    train_class_images: dict[str, list[str]] = {}
    for cls_name in class_names:
        cls_dir = os.path.join(train_dir, cls_name)
        if os.path.isdir(cls_dir):
            imgs = [
                os.path.join(cls_dir, f)
                for f in os.listdir(cls_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
            if imgs:
                train_class_images[cls_name] = imgs
    return train_class_images


def _copy_sampled_images(
    train_class_images: dict[str, list[str]], test_dir: str
) -> None:
    """Copy randomly sampled images (30-40%) from training to test directory."""
    for cls_name, imgs in train_class_images.items():
        sample_ratio = random.uniform(0.3, 0.4)
        n_samples = max(1, int(len(imgs) * sample_ratio))
        sampled_imgs = random.sample(imgs, n_samples)

        for img_path in sampled_imgs:
            ext = os.path.splitext(img_path)[1]
            rand_6_digits = f"{random.randint(0, 999999):06d}"
            new_filename = f"{cls_name}-{rand_6_digits}{ext}"
            new_filepath = os.path.join(test_dir, new_filename)
            shutil.copy(img_path, new_filepath)


def _build_test_pool(test_dir: str) -> list[tuple[str, str]]:
    """Build test pool from test directory files."""
    test_pool: list[tuple[str, str]] = []
    test_files = [
        f for f in os.listdir(test_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    for test_filename in test_files:
        true_label = test_filename.split("-")[0]
        test_filepath = os.path.join(test_dir, test_filename)
        test_pool.append((test_filepath, true_label))
    return test_pool


def _evaluate_predictions(
    test_pool: list[tuple[str, str]],
    model: MinesweeperCNN,
    device: torch.device,
    class_names: list[str],
) -> dict[str, dict[str, int]]:
    """Evaluate model predictions and collect statistics."""
    class_stats: dict[str, dict[str, int]] = {
        cls: {"total": 0, "correct": 0} for cls in class_names
    }

    for img_path, true_label in test_pool:
        pred_label = predict_image(img_path, model, device)
        if true_label in class_stats:
            class_stats[true_label]["total"] += 1
            if pred_label == true_label:
                class_stats[true_label]["correct"] += 1

    return class_stats


def _print_results(
    class_stats: dict[str, dict[str, int]],
    class_names: list[str],
    test_pool: list[tuple[str, str]],
    device: torch.device,
) -> None:
    """Print test results report."""
    print("\n" + "=" * 47 + "\n")
    print("🔍 扫雷 AI 模拟实战深度测试报告")
    print(
        f"测试规则: 随机抽取训练集 30%~40% 图片作为测试集，共抽测 {len(test_pool)} 张"
    )
    print(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"使用设备: {device}\n")

    print("=" * 47)
    print(f"| {'类别':<5} | {'测试数':>6} | {'正确数':>6} | {'准确率':>6} |")
    print(f"| {'-' * 7}-+-{'-' * 9}-+-{'-' * 9}-+-{'-' * 9} |")

    for cls_name in class_names:
        stats = class_stats[cls_name]
        acc = (stats["correct"] / stats["total"] * 100.0) if stats["total"] > 0 else 0.0
        print(
            f"| {cls_name:<7} | {stats['total']:>9} | {stats['correct']:>9} | {acc:>8.2f}% |"
        )

    print("=" * 47 + "\n")


def run_test(
    test_count: int,
    model: MinesweeperCNN,
    device: torch.device,
    root_path: Path | None = None,
) -> None:
    """纯测试逻辑：从train抖取30%~40%数据到test文件夹，模拟实战深度测试并打印报告"""
    _root = root_path or Path(__file__).resolve().parent.parent.parent.parent
    meta_path = str(_root / "model" / "minesweeper_meta.json")
    train_dir = str(_root / "src" / "dataset" / "train")
    test_dir = str(_root / "src" / "dataset" / "test")

    _, class_names = _load_meta_and_class_names(meta_path)
    if not class_names:
        return

    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir)

    train_class_images = _collect_train_images(train_dir, class_names)
    if not train_class_images:
        print("❌ 错误：未找到任何训练图片。")
        return

    random.seed()
    _copy_sampled_images(train_class_images, test_dir)
    test_pool = _build_test_pool(test_dir)
    class_stats = _evaluate_predictions(test_pool, model, device, class_names)
    _print_results(class_stats, class_names, test_pool, device)
