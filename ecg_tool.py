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
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, classification_report


# Dataset
class ECGDataset(Dataset):
    def __init__(self, img_paths, labels=None, transform=None):
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform

    def __len__(self): 
        return len(self.img_paths)

    def __getitem__(self, idx):
        image = Image.open(self.img_paths[idx]).convert('RGB')

        if self.transform:
            image = self.transform(image)

        if self.labels is not None:
            dtype = torch.long if self.labels.dtype == np.int64 else torch.float32
            return image, torch.tensor(self.labels[idx], dtype=dtype)
        return image
    
def aggregate_diagnostic_superclass(y_dic, scp_df, is_single_label):
    tmp = {}
    for key, value in y_dic.items():
        if key in scp_df.index:
            if value > 0:
                tmp[scp_df.loc[key].diagnostic_class] = value
    
    if is_single_label and len(tmp) > 1:
        sorted_tmp = sorted(tmp.items(), key=lambda item: item[1], reverse=True)
        max_class = sorted_tmp[0]
        sec_max_class = sorted_tmp[1]

        if max_class[1] == sec_max_class[1]:
            return {}
        else:
            return {max_class[0]:max_class[1]}
    
    return tmp

# 載入模型
def load_model(checkpoint_path, device):
    # 載入 checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 取得類別資訊
    class_names = checkpoint['super_classes']
    num_classes = len(class_names)
    
    # 建立模型架構（要和訓練時一樣）
    model = timm.create_model('vit_small_plus_patch16_dinov3.lvd1689m', 
                              pretrained=False, num_classes=0)
    num_features = model.num_features
    model.head = nn.Sequential(
        nn.Linear(num_features, 512),       # 第一層：擴展特徵空間
        nn.LayerNorm(512),                  # 歸一化，讓訓練更穩定
        nn.GELU(),                          # 比 ReLU 更適合 Transformer 的激活函數
        nn.Dropout(0.3),                    # 防止 20,000 筆資料過擬合
        nn.Linear(512, num_classes)         # 第二層：輸出多標籤 Logits
    )
    
    # 載入權重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"模型載入成功！")
    
    return model


def evaluate_singlelabel(model, criterion, test_loader, device, class_names):
    """單標籤多類別分類評估（softmax）"""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    running_loss = 0.0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            running_loss += loss.item() * images.size(0)
            probs = torch.softmax(logits, dim=1)
            all_preds.append(torch.argmax(logits, dim=1).cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    probs = np.concatenate(all_probs)

    true_one_hot = np.eye(len(class_names))[labels]
    
    accuracy = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average='macro', zero_division=0)
    auroc_macro = roc_auc_score(true_one_hot, probs, average='macro')
    model_report = classification_report(labels, preds, target_names=class_names, zero_division = 0, output_dict=True, digits=4)
    test_loss = running_loss / len(test_loader.dataset)
    
    return accuracy, f1_macro, auroc_macro, test_loss, model_report

def evaluate_multilabel(model, criterion, test_loader, device, class_names, threshold=0.5):
    """多標籤分類評估（sigmoid）"""
    model.eval()
    all_probs, all_preds, all_labels = [], [], []
    running_loss = 0.0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            if criterion is not None:
                loss = criterion(logits, labels)
                running_loss += loss.item() * images.size(0)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())
            all_preds.append((probs > threshold).cpu().numpy())
            all_labels.append((labels > 0.5).cpu().numpy())

    probs = np.vstack(all_probs)
    preds = np.vstack(all_preds)
    labels = np.vstack(all_labels)

    accuracy = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average='macro', zero_division=0)
    auroc_macro = roc_auc_score(labels, probs, average='macro')
    model_report = classification_report(labels, preds, target_names=class_names, zero_division = 0, output_dict=True, digits=4)
    test_loss = running_loss / len(test_loader.dataset)

    print("各類別 AUROC:")
    for i, name in enumerate(class_names):
        auroc = roc_auc_score(labels[:, i], probs[:, i])
        pos = labels[:, i].sum()
        print(f"  {name:6s}: AUROC={auroc:.4f}  正樣本={int(pos)}/{len(labels)} ({pos/len(labels)*100:.1f}%)")

    return accuracy, f1_macro, auroc_macro, test_loss, model_report

def prepare_data(is_single_label, class_names=None):
    # Load scp_statements.csv
    scp_df = pd.read_csv('ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/scp_statements.csv', index_col=0)
    scp_df = scp_df[scp_df.diagnostic == 1]

    # Load database
    df = pd.read_csv('ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/ptbxl_database.csv', index_col='ecg_id')
    df.scp_codes = df.scp_codes.apply(ast.literal_eval)
    df['diagnostic_superclass'] = df.scp_codes.apply(
        aggregate_diagnostic_superclass, scp_df=scp_df, is_single_label=is_single_label
    )
    df = df[df.diagnostic_superclass.map(len) > 0]

    if class_names is None:
        class_names = ['NORM', 'MI', 'STTC', 'CD', 'HYP']

    # Load image paths
    img_paths = [
        Path("vit_ecg_images/hr/cwt") / f"{Path(p).stem}_12lead_vit_cwt.png"
        for p in df['filename_hr']
    ]

    # Load labels（根據模式不同）
    if is_single_label:
        class_to_idx = {c: i for i, c in enumerate(class_names)}
        labels = np.array([
            class_to_idx[list(d.keys())[0]]
            for d in df['diagnostic_superclass']
        ], dtype=np.int64)
    else:
        labels = np.array([
            [d.get(name, 0) / 100.0 for name in class_names]
            for d in df['diagnostic_superclass']
        ], dtype=np.float32)

    # 使用和訓練時相同的分割
    train_img_paths, test_img_paths, train_labels, test_labels = train_test_split(
        img_paths, labels, test_size=0.1, random_state=42
    )

    print(f"資料準備完成！")
    print(f"   模式: {'單標籤' if is_single_label else '多標籤'}")
    print(f"   訓練樣本數: {len(train_img_paths)}")
    print(f"   測試樣本數: {len(test_img_paths)}")
    print(f"   標籤 shape: {test_labels.shape}\n")

    return train_img_paths, test_img_paths, train_labels, test_labels

def train_model(model, device, criterion, train_loader, num_epochs, freeze_epochs):
    # 凍結 backbone
    for name, param in model.named_parameters():
        if "head" not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    # 建立 optimizer，只更新 head
    optimizer = optim.AdamW([
        {'params': model.head.parameters(), 'lr': 5e-4},
    ], weight_decay=0)

    # 建立初始 scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=freeze_epochs)

    running_loss = 0.0

    # 訓練模型
    for epoch in range(num_epochs):
        # 第 6 回合開始解凍 backbone
        if epoch == freeze_epochs:
            print("Unfreezing backbone")
            for param in model.parameters():
                param.requires_grad = True  # 全部解凍

            # 重新建立 optimizer，分層lr
            optimizer = optim.AdamW([
                {'params': model.patch_embed.parameters(), 'lr': 1e-5},
                {'params': model.blocks.parameters(), 'lr': 1e-5},
                {'params': model.norm.parameters(), 'lr': 1e-5},
                {'params': model.head.parameters(), 'lr': 5e-5},
            ], weight_decay=0.01)

            # 重新建立針對剩餘回合的 scheduler
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs - freeze_epochs)

        model.train()
        running_loss = 0.0

        for batch_images, batch_labels in train_loader:
            batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)

            optimizer.zero_grad()
            logits = model(batch_images)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_images.size(0)
        
        # 更新學習率
        scheduler.step()

        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {running_loss / len(train_loader.dataset):.4f}")
    
    return running_loss / len(train_loader.dataset)