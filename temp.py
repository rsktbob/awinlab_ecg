import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import torchvision.models as models
import pandas as pd
from sklearn.metrics import accuracy_score
import timm
from pathlib import Path
import ast
from sklearn.model_selection import train_test_split
import numpy as np
import wfdb


# Dataset
class ECGDataset(Dataset):
    def __init__(self, img_paths, labels=None):
        self.img_paths = img_paths
        self.labels = labels
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5])
        ])
    def __len__(self): 
        return len(self.img_paths)
    def __getitem__(self, i):
        img = Image.open(self.img_paths[i]).convert('RGB')
        img = self.transform(img)
        if self.labels is not None:
            return img, torch.tensor(self.labels[i], dtype=torch.float32)
        return img



# Load scp_statements.csv for diagnostic aggregation
agg_df = pd.read_csv('ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/scp_statements.csv', index_col=0)
agg_df = agg_df[agg_df.diagnostic == 1]


def parse_scp_codes_with_probs(s):
    d = ast.literal_eval(s)
    d = {k: v for k, v in d.items() if v > 0}
    return d

def aggregate_diagnostic_superclass(y_dic):
    tmp = []
    for key, value in y_dic.items():
        if key in agg_df.index:
            if value > 0:
                tmp.append((agg_df.loc[key].diagnostic_class, value))
    return list(set(tmp))


# load and convert annotation data
df = pd.read_csv('ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/ptbxl_database.csv', index_col='ecg_id')
df.scp_codes = df.scp_codes.apply(lambda x: ast.literal_eval(x))


# Apply diagnostic superclass
df['diagnostic_superclass'] = df.scp_codes.apply(aggregate_diagnostic_superclass)
df = df[df.diagnostic_superclass.map(lambda x: len(x)) > 0]
super_classes = ['NORM', 'MI', 'STTC', 'CD', 'HYP']

# Load x data path
img_paths = []
for path in df['filename_hr']:
    img_paths.append(Path("vit_ecg_images/hr/cwt") / f"{Path(path).stem}_12lead_vit_cwt.png")

# Load y
labels_prob = np.array([
        [d.get(name, 0)/100.0 for name in super_classes] for d in df['diagnostic_superclass']
    ], dtype=np.float32)

train_paths, test_paths, train_labels, test_labels = train_test_split(
    img_paths, labels_prob, test_size=0.1, random_state=42
)

train_dataset = ECGDataset(train_paths, labels=train_labels)
test_dataset = ECGDataset(test_paths, labels=test_labels)

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)

# 模型
num_classes = len(super_classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = timm.create_model('vit_small_plus_patch16_dinov3.lvd1689m', pretrained=True, num_classes=0)
num_features = model.num_features
model.head = nn.Linear(num_features, num_classes)
model.to(device)


# 損失與優化
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
torch.backends.cudnn.benchmark = True


# 訓練
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
    print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {running_loss / len(train_dataset):.4f}")

    # 測試
    model.eval()
    test_loss = 0.0
    all_probs = []
    all_preds = []
    all_labels_list = []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * imgs.size(0)
            probs = torch.sigmoid(outputs)

            all_probs.append(probs.cpu().numpy())
            all_preds.append((probs > 0.5).cpu().numpy())
            all_labels_list.append((labels > 0.5).cpu().numpy())

    test_loss /= len(test_dataset)
    all_preds = np.vstack(all_preds)
    all_labels_list = np.vstack(all_labels_list)
    accuracy = accuracy_score(all_labels_list, all_preds)

    print(f"Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}")

    # 保存模型
    save_dir = Path("ecg_models")
    save_dir.mkdir(exist_ok=True)
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': running_loss / len(train_dataset),
        'test_loss': test_loss,
        'test_accuracy': accuracy,
        'super_classes': super_classes,
    }, save_dir / 'vit_small_plus_dinov3_30_v3.pth')
    print(f"Model saved at {save_dir / 'vit_small_plus_dinov3_30_v3.pth'}")