import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import pandas as pd
import timm
from pathlib import Path
import ast
from sklearn.model_selection import train_test_split
import numpy as np
import wfdb
from sklearn.metrics import accuracy_score
from timm.data import create_transform
from ecg_tool import ECGDataset, aggregate_diagnostic_superclass, prepare_data, evaluate_singlelabel, train_model



if __name__ == "__main__":
        # 資料類別
    class_names = ['NORM', 'MI', 'STTC', 'CD', 'HYP']

    # 模型
    num_classes = len(class_names)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model('vit_small_plus_patch16_dinov3.lvd1689m', 
                              pretrained=True, 
                              num_classes=0,
                              drop_rate=0.3,
                              attn_drop_rate=0.1)
    num_features = model.num_features
    model.head = nn.Sequential(
        nn.Linear(num_features, 512),       # 第一層：擴展特徵空間
        nn.LayerNorm(512),                  # 歸一化，讓訓練更穩定
        nn.GELU(),                          # 比 ReLU 更適合 Transformer 的激活函數
        nn.Dropout(0.3),                    # 防止 20,000 筆資料過擬合
        nn.Linear(512, num_classes)         # 第二層：輸出多標籤 Logits
    )
    model.to(device)

    # 預訓練模型的數據
    data_config = model.pretrained_cfg

    
    train_transform = create_transform(
        input_size=data_config['input_size'],
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        crop_pct=data_config.get('crop_pct', None),
        crop_mode=data_config.get('crop_mode', None),
        is_training=True,
        hflip=0.0,
        vflip=0.0,
        color_jitter=0.0,
        auto_augment=None,
    )

    test_transform = create_transform(
        input_size=data_config['input_size'],
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        crop_pct=data_config.get('crop_pct', None),
        crop_mode=data_config.get('crop_mode', None),
        is_training=False
    )

    # 訓練/測試切分
    train_img_paths, test_img_paths, train_labels, test_labels = prepare_data(is_single_label=True, class_names=class_names)

    # 準備資料集
    train_dataset = ECGDataset(
        train_img_paths,
        labels=train_labels,
        transform=train_transform
    )

    test_dataset = ECGDataset(
        test_img_paths,
        labels=test_labels,
        transform=test_transform
    )

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)


    # 損失與優化
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.05)
    torch.backends.cudnn.benchmark = True

    # 訓練
    num_epochs = 20
    freeze_epochs = 5

    train_loss = train_model(model, device, criterion, train_loader, num_epochs, freeze_epochs)

    # 測試
    test_loss = evaluate_singlelabel(model, criterion, test_loader, device, class_names)

    # 保存模型
    save_dir = Path("ecg_models")
    save_dir.mkdir(exist_ok=True)
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'test_loss': test_loss,
        'super_classes': class_names,
    }, save_dir / 'vit_small_plus_dinov3_30_v6.pth')
