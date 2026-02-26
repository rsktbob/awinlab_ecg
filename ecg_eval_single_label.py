import torch
from torch import nn
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score,classification_report
from torch.utils.data import DataLoader
import timm
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset
import pandas as pd
import ast
from ecg_tool import ECGDataset, aggregate_diagnostic_superclass
from timm.data import create_transform
from ecg_tool import load_model, evaluate_singlelabel, prepare_data

if __name__ == "__main__":
    # 設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = "ecg_models/vit_small_plus_dinov3_30_v6.pth"
    
    print(f"使用裝置: {device}\n")
    
    # 1. 載入模型
    model, class_names = load_model(checkpoint_path, device)
    
    # 2. 準備測試數據
    _, test_img_paths, _, test_labels = prepare_data(is_single_label=True, class_names=class_names)
    
    # 3. 建立 DataLoader
    data_config = model.pretrained_cfg
    test_transform = create_transform(
        input_size=data_config['input_size'],
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        crop_pct=data_config.get('crop_pct', None),
        crop_mode=data_config.get('crop_mode', None),
        is_training=False
    )

    test_dataset = ECGDataset(test_img_paths, labels=test_labels, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, 
                            num_workers=0, pin_memory=True)
    
    # 4. 評估模型
    results = evaluate_singlelabel(model, test_loader, device, class_names)
    
    print("\n✅ 評估完成！")