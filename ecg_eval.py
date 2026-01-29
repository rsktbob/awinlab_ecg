import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms as T
from PIL import Image
import timm
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, hamming_loss, 
    precision_score, recall_score, roc_auc_score
)
from pathlib import Path
import pandas as pd
import numpy as np
from ecg_train import parse_scp_codes_with_probs, ECGDataset

# --- 預測函數 ---
def predict_ecg(img_path, model, label_names, device, threshold=0.5):
    img = Image.open(img_path).convert("RGB")
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.sigmoid(output).squeeze(0)

    pred_indices = (probs >= threshold).nonzero(as_tuple=True)[0]
    pred_labels = [label_names[i] for i in pred_indices]
    pred_probs = probs[pred_indices].cpu().numpy()
    return pred_labels, pred_probs

if __name__ == "__main__":
    # --- 載入模型 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load("ecg_models/vit_small_plus_dinov3_30.pth", map_location=device, weights_only=False)

    label_names = checkpoint['label_names']
    num_classes = len(label_names)

    model = timm.create_model('vit_small_plus_patch16_dinov3.lvd1689m', pretrained=True, num_classes=0)
    num_features = model.num_features
    model.head = nn.Linear(num_features, num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    # 標籤與機率向量
    df = pd.read_csv('ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/ptbxl_database.csv')
    all_dicts = [parse_scp_codes_with_probs(s) for s in df['scp_codes']]
    all_names = sorted({k for d in all_dicts for k in d})
    labels_prob = np.array([
        [d.get(name, 0)/100.0 for name in all_names] for d in all_dicts
    ], dtype=np.float32)
    print(all_names)
    all_name_pos = {name : i for i, name in enumerate(all_names)}

    # 資料分割
    img_dir = Path("vit_ecg_images")
    img_paths = [
        p for p in img_dir.glob("**/*12lead*.png")
        if "cwt" in p.name
    ]
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        img_paths, labels_prob, test_size=0.1, random_state=42
    )

    train_dataset = ECGDataset(train_paths, labels=train_labels)
    test_dataset = ECGDataset(test_paths, labels=test_labels)

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)

    model.eval()
    all_probs = []
    all_preds = []
    all_labels_list = []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            probs = torch.sigmoid(outputs)
            
            all_probs.append(probs.cpu().numpy())
            all_preds.append((probs > 0.5).cpu().numpy())
            all_labels_list.append((labels > 0.5).cpu().numpy())

    all_probs = np.vstack(all_probs)
    all_preds = np.vstack(all_preds)
    all_labels_list = np.vstack(all_labels_list)

    # 計算精準度
    accuracy = accuracy_score(all_labels_list, all_preds)

    # 計算f1 score
    f1_macro = f1_score(all_labels_list, all_preds, average='macro', zero_division=0)

    valid_labels = []
    for i in range(all_labels_list.shape[1]):
        if len(np.unique(all_labels_list[:, i])) > 1:
            valid_labels.append(i)

    # 計算auroc
    auroc_macro = roc_auc_score(
        all_labels_list[:, valid_labels], 
        all_probs[:, valid_labels], 
        average='macro'
    )

    # 輸出結果
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-Score (Macro):  {f1_macro:.4f}")
    print(f"AUROC (Macro):     {auroc_macro:.4f}")