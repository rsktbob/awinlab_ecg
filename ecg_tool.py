import torch
from torch import nn, optim
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
import pandas as pd
import timm
from copy import deepcopy
from pathlib import Path
import ast
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import (
    f1_score, roc_auc_score, precision_score,
    recall_score, classification_report
)
import random
import matplotlib.pyplot as plt
import os

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


# Augmentation
class RandomShrinkSignal:
    def __init__(self, num_signals=12, signal_h=16, width=224, scale=(0.8, 1.0)):
        self.num_signals = num_signals
        self.signal_h    = signal_h
        self.width       = width
        self.scale       = scale

    def __call__(self, img):
        img     = TF.to_tensor(img)
        signals = []

        for i in range(self.num_signals):
            start  = i * self.signal_h
            end    = (i + 1) * self.signal_h
            signal = img[:, start:end, :]

            s     = random.uniform(*self.scale)
            new_h = int(self.signal_h * s)
            new_w = int(self.width * s)

            resized = TF.resize(signal, [new_h, new_w])
            canvas  = torch.zeros_like(signal)

            y = random.randint(0, self.signal_h - new_h)
            x = random.randint(0, self.width - new_w)
            canvas[:, y:y + new_h, x:x + new_w] = resized
            signals.append(canvas)

        output = torch.cat(signals, dim=1)
        output = TF.resize(output, [384, 384],
                           interpolation=TF.InterpolationMode.BICUBIC)
        return output

# 畫attention map
def draw_attention_img(img, tensor, model, save_path):
    # ---- hook 所有層 ----
    all_attentions = []

    def make_hook(attn_module):
        def hook_fn(module, input):
            x = input[0].detach()
            B, N, C = x.shape
            qkv = attn_module.qkv(x)
            num_heads = attn_module.num_heads
            head_dim = C // num_heads
            scale = head_dim ** -0.5
            qkv = qkv.reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
            q, k, _ = qkv.unbind(0)
            attn_w = (q @ k.transpose(-2, -1)) * scale
            attn_w = attn_w.softmax(dim=-1)  # (1, num_heads, N, N)
            all_attentions.append(attn_w[0].mean(dim=0).detach())  # (N, N)
        return hook_fn

    hooks = []
    for block in model.backbone.blocks:
        h = block.attn.register_forward_pre_hook(make_hook(block.attn))
        hooks.append(h)

    with torch.no_grad():
        _ = model(tensor)

    for h in hooks:
        h.remove()

    # ---- Attention Rollout ----
    # 每層加上 residual connection 再連乘
    N = all_attentions[0].shape[0]
    result = torch.eye(N, device=tensor.device)

    for attn in all_attentions:
        # 加上 residual (identity)，模擬 skip connection
        attn_with_residual = attn + torch.eye(N, device=tensor.device)
        # normalize
        attn_with_residual = attn_with_residual / attn_with_residual.sum(dim=-1, keepdim=True)
        result = attn_with_residual @ result

    # CLS token 對所有 patch 的最終 attention
    cls_attn = result[0, 1:]  # (576,)
    cls_attn = (cls_attn - cls_attn.min()) / (cls_attn.max() - cls_attn.min())

    grid_size = int(cls_attn.shape[0] ** 0.5)  # 24
    attn_map = cls_attn.detach().cpu().numpy().reshape(grid_size, grid_size)

    w, h = img.size
    attn_resized = np.array(
        Image.fromarray(attn_map).resize((w, h), Image.BILINEAR)
    )

    # ---- overlay ----
    img_np = np.array(img).astype(np.float32) / 255.0
    cmap = plt.get_cmap('jet')
    attn_colored = cmap(attn_resized)[..., :3]
    overlay = np.clip((0.5 * img_np + 0.5 * attn_colored) * 255, 0, 255).astype(np.uint8)

    Image.fromarray(overlay).save(save_path)


# 預測圖片類別
def predict_img(model, device, test_transform, img_path, is_attention_map):
    img_path = Path(img_path)
    img_idx = img_path.stem[:5]
    save_folder = Path('ecg_attention_images/') / img_idx

    img = Image.open(img_path).convert('RGB')
    tensor = test_transform(img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

    class_probs = [
        (class_name, prob)
        for class_name, prob in zip(SUPER_CLASS_NAMES, probs)
        if prob > 0.5
    ]

    if len(class_probs) > 0:
        os.makedirs(save_folder, exist_ok=True)
    else:
        return class_probs

    if is_attention_map:
        save_path = os.path.join(
            save_folder, f"{img_idx}_attn.png"
        )

        draw_attention_img(
            img=img,
            tensor=tensor,
            model=model,
            save_path=save_path
        )

        print(f'{img_idx}_attn saved')

    return class_probs

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
        IMAGE_ROOT  / f"{Path(p).stem}_12lead_vit_cwt.png"
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
def load_model(model_path, model_name, device):
    checkpoint = torch.load(model_path, map_location=device)
    super_classes = checkpoint.get('super_classes', [])
    num_classes = len(super_classes)
    model = ECGModel(model_name=model_name, num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    return model


# 訓練一回合
def train_one_epoch(model, loader, device, criterion, optimizer):
    model.train()
    running_loss = 0.0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss * images.size(0)

    return (running_loss / len(loader.dataset)).item()


# 評估一回合
def eval_one_epoch(model, loader, device, criterion):
    model.eval()
    running_loss = torch.tensor(0.0, device=device)
    all_probs, all_labels = [], []

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            running_loss += criterion(logits, labels) * images.size(0)
            all_probs.append(torch.sigmoid(logits))
            all_labels.append(labels)

    avg_loss   = (running_loss / len(loader.dataset)).item()
    all_probs  = torch.cat(all_probs).cpu().numpy()
    all_labels = torch.cat(all_labels).cpu().numpy()
    return avg_loss, all_probs, all_labels


# 評估模型
def evaluate_model(model, criterion, test_loader, device, class_names, threshold=0.5):
    test_loss, probs, labels = eval_one_epoch(model, test_loader, device, criterion)
    labels = labels.astype(int)
    preds  = (probs > threshold).astype(int)

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

# 選擇適合學習率
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
    ], weight_decay=0.05)
    # return optim.AdamW([
    #     {'params': model.parameters(), 'lr': 5e-4},
    # ], weight_decay=0)


# 訓練
def train_model(model, device, criterion, train_loader, test_loader, 
                num_epochs, freeze_epochs):

    for p in model.backbone.parameters():
        p.requires_grad = False

    optimizer = _build_optimizer(model, frozen=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=freeze_epochs)
    train_loss_list, test_loss_list = [], []

    best_loss = float('inf')
    best_model_wts = deepcopy(model.state_dict())
    counter = 0

    for epoch in range(num_epochs):

        if epoch == freeze_epochs:
            print("Unfreezing backbone")
            for p in model.parameters():
                p.requires_grad = True
            optimizer = _build_optimizer(model, frozen=False)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=num_epochs - freeze_epochs)

            # warmup_epochs = int(0.1*(num_epochs - freeze_epochs))

            # print(warmup_epochs)

            # # Warmup：從 lr * start_factor 線性升到 lr
            # warmup_scheduler = LinearLR(
            #     optimizer,
            #     start_factor=0.1,
            #     end_factor=1.0,
            #     total_iters=warmup_epochs
            # )

            # # Cosine Annealing：warmup 結束後開始退火
            # cosine_scheduler = CosineAnnealingLR(
            #     optimizer,
            #     T_max=num_epochs - freeze_epochs - warmup_epochs,
            #     eta_min=1e-6
            # )

            # # 串接
            # scheduler = SequentialLR(
            #     optimizer,
            #     schedulers=[warmup_scheduler, cosine_scheduler],
            #     milestones=[warmup_epochs]
            # )


        train_loss = train_one_epoch(model, train_loader, device, criterion, optimizer)
        train_loss_list.append(train_loss)

        test_loss, _, _ = eval_one_epoch(model, test_loader, device, criterion)
        test_loss_list.append(test_loss)
        
        lrs = [f"{g['lr']:.2e}" for g in optimizer.param_groups]
        print(f"Epoch [{epoch+1:>3}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f}  "
              f"Test Loss: {test_loss:.4f}  "
              f"LRs: {lrs}  ")

        scheduler.step()
        
        # Early Stopping 判斷
        if test_loss < best_loss:
            best_loss = test_loss
            counter = 0
            best_model_wts = deepcopy(model.state_dict())   # 存最佳模型
        else:
            counter += 1

        if counter >= 5:
            print("Early stopping triggered!")
            break

    # 回復最佳模型
    if best_model_wts:
        model.load_state_dict(best_model_wts)

    return train_loss_list, test_loss_list