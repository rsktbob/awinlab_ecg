import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as T
import timm
from pathlib import Path
import numpy as np
from timm.data import create_transform
from ecg_tool import ECGDataset, ECGModel, prepare_data, evaluate_model, train_model

if __name__ == "__main__":
    # 資料類別
    class_names = ['NORM', 'MI', 'STTC', 'CD', 'HYP']

    # 模型
    num_classes = len(class_names)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ECGModel(model_name='deit3_small_patch16_384.fb_in1k',num_classes=num_classes)
    model.to(device)
    

    print(model.head)

    # 預訓練模型的數據
    data_config = timm.data.resolve_model_data_config(model.backbone)

    train_transform = create_transform(
        **data_config,
        is_training=True,
        hflip=0.0,
        vflip=0.0,
        color_jitter=0,
        scale=(0.6, 1.0), 
        auto_augment=None,
    )

    test_transform = create_transform(
        **data_config,
        is_training=False,
    )
    
    # 訓練/測試切分
    train_img_paths, test_img_paths, train_labels, test_labels = prepare_data(class_names=class_names)

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
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=6, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=6, pin_memory=True,  persistent_workers=True)

    # 計算 pos_weight
    labels_np = np.array(train_labels) 
    pos = labels_np.sum(axis=0)
    neg = labels_np.shape[0] - pos
    pos_weight = torch.sqrt(torch.tensor(neg / (pos + 1e-6), dtype=torch.float32)).to(device)

    print("pos_weight per class:", pos_weight)

    # 損失與優化
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    torch.backends.cudnn.benchmark = True

    # 訓練
    num_epochs = 16
    freeze_epochs = 3

    train_loss_list, test_loss_list = train_model(model, device, criterion, train_loader, test_loader, num_epochs, freeze_epochs)

    # 測試
    precision_micro, recall_micro, f1_micro, precision_macro, recall_macro, f1_macro, auroc_macro, test_loss, model_report = evaluate_model(model, criterion, test_loader, device, class_names)
    print(f"最終測試損失: {test_loss:.4f}")
 
    # 保存模型
    save_dir = Path("ecg_models")
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'train_loss_list': train_loss_list,
        'test_loss_list': test_loss_list,
        'super_classes': class_names,
        'precision_micro': precision_micro,
        'recall_micro': recall_micro,
        'f1_micro': f1_micro,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'auroc_macro': auroc_macro,
        'model_report': model_report
    }, save_dir / 'deit3_small_patch16_384_16_v3.pth')
