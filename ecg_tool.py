import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.v2 as T
import pandas as pd
import timm
from pathlib import Path
import ast
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score,f1_score, roc_auc_score, precision_score, recall_score, classification_report
import torch
import random
from PIL import Image
import torchvision.transforms.functional as TF

SUPER_CLASS_NAMES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
PTB_XL_ROOT = Path('ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3')
IMAGE_ROOT   = Path('vit_ecg_images/hr/cwt')


# Dataset
class ECGDataset(Dataset):
    def __init__(self, img_paths, labels=None, transform=None):
        self.img_paths = img_paths
        self.labels    = torch.tensor(labels, dtype=torch.float32)
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        image = Image.open(self.img_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx] 


# Model
class ECGModel(nn.Module):
    def __init__(self, model_name, num_classes, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.head = nn.Linear(self.backbone.num_features, num_classes)
    
    def forward(self, x):
        features = self.backbone(x)
        logits = self.head(features)
        return logits

class RandomShrinkSignal:
    def __init__(self, num_signals=12, signal_h=16 , width=224, scale=(0.8, 1.0)):
        self.num_signals = num_signals
        self.signal_h = signal_h
        self.width = width
        self.scale = scale

    def __call__(self, img):

        img = TF.to_tensor(img)
        signals = []

        for i in range(self.num_signals):

            start = i * self.signal_h
            end = (i + 1) * self.signal_h

            signal = img[:, start:end, :]

            s = random.uniform(*self.scale)

            new_h = int(self.signal_h * s)
            new_w = int(self.width * s)

            resized = TF.resize(signal, [new_h, new_w])

            canvas = torch.zeros_like(signal)

            y = random.randint(0, self.signal_h - new_h)
            x = random.randint(0, self.width - new_w)

            canvas[:, y:y+new_h, x:x+new_w] = resized

            signals.append(canvas)

        output = torch.cat(signals, dim=1)              
        output = TF.resize(output, [384, 384],
                       interpolation=TF.InterpolationMode.BICUBIC)

        return output

# 資料準備
def prepare_data(class_names=SUPER_CLASS_NAMES):

    def aggregate_diagnostic_superclass(y_dic, scp_df):
        return {
            scp_df.loc[key].diagnostic_class: value
            for key, value in y_dic.items()
            if key in scp_df.index and value > 0
        }
    
    scp_df = pd.read_csv(PTB_XL_ROOT / 'scp_statements.csv', index_col=0)
    scp_df = scp_df[scp_df.diagnostic == 1]

    df = pd.read_csv(PTB_XL_ROOT / 'ptbxl_database.csv', index_col='ecg_id')
    df.scp_codes = df.scp_codes.apply(ast.literal_eval)
    df['diagnostic_superclass'] = df.scp_codes.apply(
        aggregate_diagnostic_superclass, scp_df=scp_df
    )
    df = df[df.diagnostic_superclass.map(len) > 0]

    img_paths = [
        IMAGE_ROOT / f"{Path(p).stem}_12lead_vit_cwt.png"
        for p in df['filename_hr']
    ]
    labels = np.array([
        [d.get(name, 0) / 100.0 for name in class_names]
        for d in df['diagnostic_superclass']
    ], dtype=np.float32)

    train_paths, test_paths, train_labels, test_labels = train_test_split(
        img_paths, labels, test_size=0.1, random_state=42
    )

    print(f"資料準備完成！訓練={len(train_paths)}  測試={len(test_paths)}  標籤shape={test_labels.shape}\n")
    return train_paths, test_paths, train_labels, test_labels

# 模型載入
def load_model(model_path, model_name, num_classes, device):
    model = ECGModel(model_name=model_name, num_classes=num_classes)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

def predict_img(model, img):
    super_class = model.get('super_classes', 0)
    
    with torch.no_grad():
        logits = model(img)             
        probs = torch.sigmoid(logits)   
        probs = probs.squeeze(0)
        
        class_probs = [(c, float(p)) for c, p in zip(super_class, probs)]
    
    return class_probs

# 訓練一回合
def train_one_epoch(model, loader, device, criterion, optimizer):
    model.train()
    running_loss = 0.0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss * images.size(0)

    avg_loss = (running_loss / len(loader.dataset)).item()

    return avg_loss


# 評估一回合
def eval_one_epoch(model, loader, device, criterion):
    model.eval()
    running_loss = torch.tensor(0.0, device=device)  # 留在 GPU，避免每次 sync
    all_probs, all_labels = [], []

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, labels)

            running_loss += loss * images.size(0)
            all_probs.append(torch.sigmoid(logits))
            all_labels.append(labels)

    avg_loss = (running_loss / len(loader.dataset)).item()
    all_probs = torch.cat(all_probs).cpu().numpy()
    all_labels = torch.cat(all_labels).cpu().numpy()

    return avg_loss, all_probs, all_labels


# 評估
def evaluate_model(model, criterion, test_loader, device, class_names, threshold=0.5):
    test_loss, probs, labels = eval_one_epoch(model, test_loader, device, criterion)
    labels = labels.astype(int)
    preds = (probs > threshold).astype(int)


    precision_micro = precision_score(labels, preds, average='micro', zero_division=0)
    recall_micro    = recall_score   (labels, preds, average='micro', zero_division=0)
    f1_micro        = f1_score       (labels, preds, average='micro', zero_division=0)
    precision_macro = precision_score(labels, preds, average='macro', zero_division=0)
    recall_macro    = recall_score   (labels, preds, average='macro', zero_division=0)
    f1_macro        = f1_score       (labels, preds, average='macro', zero_division=0)
    auroc_macro     = roc_auc_score  (labels, probs, average='macro')
    report          = classification_report(
                        labels, preds, target_names=class_names,
                        zero_division=0, output_dict=True, digits=4)

    print("各類別 AUROC:")
    for i, name in enumerate(class_names):
        auroc = roc_auc_score(labels[:, i], probs[:, i])
        pos   = int(labels[:, i].sum())
        print(f"  {name:6s}: AUROC={auroc:.4f}  正樣本={pos}/{len(labels)} ({pos/len(labels)*100:.1f}%)")

    return precision_micro, recall_micro, f1_micro, precision_macro, recall_macro, f1_macro, auroc_macro, test_loss, report


# 學習率選擇
def _build_optimizer(model, frozen):
    if frozen:
        return optim.AdamW([
            {'params': model.head.parameters(), 'lr': 1e-3},
        ], weight_decay=0)
    return optim.AdamW([
        {'params': model.backbone.patch_embed.parameters(), 'lr': 3e-6},
        {'params': model.backbone.norm.parameters(),        'lr': 3e-5},
        {'params': model.backbone.blocks.parameters(),      'lr': 3e-5},
        {'params': model.head.parameters(),                 'lr': 5e-4},
    ], weight_decay=0)


# 訓練
def train_model(model, device, criterion, train_loader, test_loader, num_epochs, freeze_epochs):
    # Phase 1：凍結 backbone
    for p in model.backbone.parameters():
        p.requires_grad = False

    optimizer = _build_optimizer(model, frozen=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=freeze_epochs)

    train_loss_list, test_loss_list = [], []

    for epoch in range(num_epochs):
        # Phase 2：解凍 backbone
        if epoch == freeze_epochs:
            print("Unfreezing backbone")
            for p in model.parameters():
                p.requires_grad = True
            optimizer = _build_optimizer(model, frozen=False)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=num_epochs - freeze_epochs
            )

        # ── train ──
        train_loss = train_one_epoch(model, train_loader, device, criterion, optimizer)
        train_loss_list.append(train_loss)

        # ── eval ──
        test_loss, _, _ = eval_one_epoch(model, test_loader, device, criterion)
        test_loss_list.append(test_loss)

        scheduler.step()

        lrs = [f"{g['lr']:.2e}" for g in optimizer.param_groups]
        print(f"Epoch [{epoch+1:>3}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f}  "
              f"Test Loss: {test_loss:.4f}  "
              f"LRs: {lrs}")

    return train_loss_list, test_loss_list    