import torch
from torch import nn
from torchvision import models, transforms as T
from PIL import Image
import timm
from pathlib import Path
import pandas as pd
import numpy as np

# --- 預測函數 ---
def predict_ecg(img_path, model, label_names, device, img_size=224, threshold=0.5):
    img = Image.open(img_path).convert("L")
    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5])
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.sigmoid(output).squeeze(0)

    pred_indices = (probs >= threshold).nonzero(as_tuple=True)[0]
    pred_labels = [label_names[i] for i in pred_indices]
    pred_probs = probs[pred_indices].cpu().numpy()
    return pred_labels, pred_probs

# --- 載入模型 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load("saved_models/ecg_vit_model_v2.pth", map_location=device,  weights_only=False)

label_names = checkpoint['label_names']
num_classes = len(label_names)

model = timm.create_model(
    'vit_small_patch16_dinov3.lvd1689m',  # 你訓練時用的 EVA
    pretrained=False,
    in_chans=1,  # 單通道
    num_classes=num_classes
)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# --- 資料 ---
img_dir = Path("vit_ecg_images")
img_paths = sorted(img_dir.glob("**/*12lead*.png"))
df = pd.read_csv("ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/ptbxl_database.csv")

# --- 預測 ---
scp_series = df['scp_codes']

for expert_label, img_path in zip(scp_series, img_paths):
    pred_labels, pred_probs = predict_ecg(img_path, model, label_names, device)
    pred_dict = {label: round(float(prob), 4) for label, prob in zip(pred_labels, pred_probs)}
    print(f"{img_path.name} 預測: {pred_dict} 專家: {expert_label}")
