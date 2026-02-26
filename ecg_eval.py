import torch
from torch import nn
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, classification_report
from torch.utils.data import DataLoader
import timm
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset
import json
import pandas as pd
import ast
from ecg_tool import ECGDataset, aggregate_diagnostic_superclass
from timm.data import create_transform
from ecg_tool import load_model, evaluate_multilabel, prepare_data

if __name__ == "__main__":
    # 1. 載入模型
    model_path = Path("ecg_models/vit_small_plus_dinov3_30_v6.pth")
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # 2. 印出基礎資訊
    print(f"訓練回合數 (Epoch): {checkpoint.get('epoch', 'N/A')}")
    print(f"最終訓練損失 (Train Loss): {checkpoint.get('train_loss', 0):.4f}")
    print(f"最終測試損失 (Test Loss): {checkpoint.get('test_loss', 0):.4f}")
    print("-" * 30)

    # 3. 印出核心指標
    print(f"整體 Accuracy: {checkpoint.get('accuracy', 0):.4f}")
    print(f"整體 Macro F1: {checkpoint.get('f1_macro', 0):.4f}")
    print(f"整體 Macro AUROC: {checkpoint.get('auroc_macro', 0):.4f}")
    print("-" * 30)

    # 4. 印出 Dict 格式的分類報告
    report = checkpoint['model_report']
    print("詳細類別指標 (Classification Report):")
    print(json.dumps(report, indent=4))
