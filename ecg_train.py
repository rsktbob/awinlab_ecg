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

# 標籤解析
def parse_scp_codes_with_probs(s):
    try:
        d = ast.literal_eval(s)
        d = {k: v for k, v in d.items() if v > 0}
        return d if d else {'NORM': 100.0}
    except:
        return {'NORM': 100.0}

def predict_accuracy(label_positions, predicted_probs, true_labels):
    threshold = 0.5
    correct_count = 0
    
    for label in true_labels:
        pos = label_positions[label]
        predicted_value = predicted_probs[pos]
        true_value = true_labels[label]
        
        predicted_class = 1 if predicted_value > threshold else 0
        true_class = 1 if true_value > threshold / 100 else 0
        
        if predicted_class == true_class:
            correct_count += 1
    
    return correct_count



if __name__ == "__main__":
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

    # 模型
    num_classes = len(all_names)
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
        'label_names': all_names,
    }, save_dir / 'vit_small_plus_dinov3_30_v2.pth')
    print(f"Model saved at {save_dir / 'vit_small_plus_dinov3_30_v2.pth'}")