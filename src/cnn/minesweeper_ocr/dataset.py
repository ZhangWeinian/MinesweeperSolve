import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

TARGET_CLASSES = ["flag", "1", "2", "3", "4", "5", "6", "7", "8"]


class MinesweeperDataset(Dataset):
    def __init__(self, root_dir: str) -> None:
        self.samples = []
        self.classes = TARGET_CLASSES
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.transform = transforms.Compose(
            [
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.samples.append((os.path.join(cls_dir, img_name), self.class_to_idx[cls_name]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        return self.transform(image) if self.transform else image, label
