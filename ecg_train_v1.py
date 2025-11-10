import wfdb
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def segment_signal(signal, patch_size=100, stride=100):
    patches = [signal[i:i+patch_size] for i in range(0, len(signal)-patch_size+1, stride)]
    return np.array(patches)

root_dir = Path("ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/records100/00000")
file_name = "00001_lr"

record = wfdb.rdrecord(str(root_dir / file_name))
signals = record.p_signal

total_patches = [segment_signal(signals[:, i]) for i in range(signals.shape[1])]

print("通道數:", len(total_patches))
print("第一個通道 patch 數量:", total_patches[0].shape[0])
print("每個 patch 長度:", total_patches[0].shape[1])


class ViT1D(nn.Module):
    def __init__(self, patch_size=100, d_model=64, nhead=8, num_layers=6, num_classes=5):
        super(ViT1D, self).__init__()
        self.patch_size = patch_size
        self.d_model = d_model

        # 將每個 patch (長度 patch_size) 投影成 d_model 維的向量
        self.linear_proj = nn.Linear(patch_size, d_model)

        # [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # 可學習的位置編碼
        self.pos_embedding = nn.Parameter(torch.randn(1, 512, d_model))  # 最多支援 512 個 patch

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 最後分類層
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        """
        x: (batch_size, num_patches, patch_size)
        """
        B, N, P = x.shape

        # (B, N, P) → (B, N, d_model)
        x = self.linear_proj(x)

        # 加上 [CLS] token
        cls_token = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
        x = torch.cat((cls_token, x), dim=1)  # (B, N+1, d_model)

        # 加上位置編碼
        x = x + self.pos_embedding[:, :N+1, :]

        # Transformer
        x = self.transformer(x)  # (B, N+1, d_model)

        # 取出 [CLS] token 向量
        cls_out = x[:, 0]

        # 分類頭
        out = self.fc(cls_out)
        return out