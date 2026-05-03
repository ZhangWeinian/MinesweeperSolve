import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from src.cnn.minesweeper_ocr.dataset import MinesweeperDataset
from src.export import MinesweeperCNN


def run_training(
    data_dir: str | None = None,
    save_dir: str | None = None,
    epochs: int = 10,
    device: torch.device = torch.device("cpu"),
    root_path: Path | None = None,
) -> tuple[str, str]:
    """执行训练流程，返回 (模型权重路径, 元数据路径)"""
    _root = root_path or Path(__file__).resolve().parent.parent.parent.parent
    _data_dir = data_dir or str(_root / "src" / "dataset" / "train")
    _save_dir = save_dir or str(_root / "model")
    os.makedirs(_save_dir, exist_ok=True)

    print(f"🔥 使用设备: {device}")
    if device.type == "cuda":
        print(f"GPU 名称: {torch.cuda.get_device_name(0)}")
        torch.cuda.reset_peak_memory_stats()
        print(f"初始显存占用: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

    print("\n📥 正在加载数据集...")
    full_dataset = MinesweeperDataset(root_dir=_data_dir)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)

    print(f"总样本数: {len(full_dataset)} | 训练集: {train_size} | 验证集: {val_size}")
    print(
        f"网络参数量: {sum(p.numel() for p in MinesweeperCNN().parameters()) / 1000:.2f} K"
    )

    model = MinesweeperCNN(num_classes=9).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    start_time = time.time()
    print(f"\n🚀 开始训练 {epochs} 个 Epoch...\n")

    history = []
    for epoch in range(epochs):
        model.train()
        running_loss, train_correct, train_total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (preds == labels).sum().item()

        train_acc = 100.0 * train_correct / train_total
        train_loss = running_loss / len(train_loader)

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
                _, preds = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (preds == labels).sum().item()

        val_acc = 100.0 * val_correct / val_total
        val_loss = val_loss / len(val_loader)
        history.append((train_loss, val_loss, train_acc, val_acc))

        print(
            f"Epoch [{epoch + 1:02d}/{epochs}] | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:5.1f}% | "
            f"Val Loss: {val_loss:.4f}准确率: {val_acc:5.1f}%"
        )

    total_time = time.time() - start_time
    if device.type == "cuda":
        peak_mem = torch.cuda.max_memory_allocated() / 1024**2
        print("\n📊 训练资源消耗统计:")
        print(f"  - 总用时: {total_time:.2f} 秒")
        print(f"  - 峰值显存: {peak_mem:.2f} MB")
        print(f"  - 平均速度: {total_time / epochs:.2f} 秒/Epoch")

    idx_to_class = {v: k for k, v in full_dataset.class_to_idx.items()}
    meta_path = os.path.join(_save_dir, "minesweeper_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(idx_to_class, f, ensure_ascii=False, indent=4)

    model_path = os.path.join(_save_dir, "minesweeper_cnn.pth")
    torch.save(model.state_dict(), model_path)
    print(f"\n✅ 模型及元数据已保存至: {_save_dir}/")

    return model_path, meta_path
