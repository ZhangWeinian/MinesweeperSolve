import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.export import MinesweeperCNN

IDX_TO_CLASS: dict[int, str] = {0: "flag", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8"}

TRANSFORM = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)


def is_pure_color(image_path: str) -> bool:
    """判断是否为纯色"""
    img = cv2.imread(image_path)
    if img is None:
        return False
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    std_dev = np.std(gray)
    return bool(std_dev < 5)


def classify_pure_color(image_path: str) -> str:
    """区分空白和盲区"""
    img = cv2.imread(image_path)
    if img is None:
        return "0"

    b, g, r = img[img.shape[0] // 2, img.shape[1] // 2]
    return "hidden" if (b > 150 and b > r and b > g) else "0"


def predict_image(image_path: str, model: MinesweeperCNN, device: torch.device) -> str:
    """完整的混合推理流程"""
    if is_pure_color(image_path):
        return classify_pure_color(image_path)

    image = Image.open(image_path).convert("RGB")
    tensor = torch.as_tensor(TRANSFORM(image))
    tensor = torch.unsqueeze(tensor, 0).to(device)

    model.eval()
    with torch.no_grad():
        _, idx = torch.max(model(tensor), 1)

    return IDX_TO_CLASS[int(idx.item())]
