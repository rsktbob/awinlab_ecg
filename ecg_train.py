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
    def __getitem__(self, idx):
        image = Image.open(self.img_paths[idx]).convert('RGB')
        image = self.transform(image)
        if self.labels is not None:
            return image, torch.tensor(self.labels[idx], dtype=torch.float32)
        return image



# Load scp_statements.csv for diagnostic aggregation
agg_df = pd.read_csv('ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/scp_statements.csv', index_col=0)
agg_df = agg_df[agg_df.diagnostic == 1]


def parse_scp_codes_with_probs(s):
    d = ast.literal_eval(s)
    d = {k: v for k, v in d.items() if v > 0}
    return d

def aggregate_diagnostic_superclass(y_dic):
    tmp = {}
    for key, value in y_dic.items():
        if key in agg_df.index:
            if value > 0:
                tmp[agg_df.loc[key].diagnostic_class] = value
    return tmp

if __name__ == "__main__":
    # load and convert annotation data
    df = pd.read_csv('ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/ptbxl_database.csv', index_col='ecg_id')
    df.scp_codes = df.scp_codes.apply(lambda x: ast.literal_eval(x))


    # Apply diagnostic superclass
    df['diagnostic_superclass'] = df.scp_codes.apply(aggregate_diagnostic_superclass)
    df = df[df.diagnostic_superclass.map(lambda x: len(x)) > 0]
    class_names = ['NORM', 'MI', 'STTC', 'CD', 'HYP']


    # Load x data path
    img_paths = []
    for path in df['filename_lr']:
        img_paths.append(Path("vit_ecg_images/lr/cwt") / f"{Path(path).stem}_12lead_vit_cwt.png")

    # Load y
    labels_prob = np.array([
            [d.get(name, 0)/100.0 for name in class_names] for d in df['diagnostic_superclass']
        ], dtype=np.float32)


    train_img_paths, test_img_paths, train_labels, test_labels = train_test_split(
        img_paths, labels_prob, test_size=0.1, random_state=42
    )

    train_dataset = ECGDataset(train_img_paths, labels=train_labels)
    test_dataset = ECGDataset(test_img_paths, labels=test_labels)

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)

    # 模型
    num_classes = len(class_names)
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
    num_epochs = 30
    for epoch in range(num_epochs):
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
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {running_loss / len(train_dataset):.4f}")

    # 測試
    model.eval()
    test_loss = 0.0
    predicted_probs = []
    predicted_binary_labels = []
    true_binary_labels = []

    with torch.no_grad():
        for batch_images, batch_labels in test_loader:
            batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)
            logits = model(batch_images)
            loss = criterion(logits, batch_labels)
            test_loss += loss.item() * batch_images.size(0)
            prob = torch.sigmoid(logits)

            predicted_probs.append(prob.cpu().numpy())
            predicted_binary_labels.append((prob > 0.5).cpu().numpy())
            true_binary_labels.append((batch_labels > 0.5).cpu().numpy())

    test_loss /= len(test_dataset)
    predicted_binary_labels = np.vstack(predicted_binary_labels)
    true_binary_labels = np.vstack(true_binary_labels)
    accuracy = accuracy_score(true_binary_labels, predicted_binary_labels)

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
        'super_classes': class_names,
    }, save_dir / 'vit_small_plus_dinov3_30_v5.pth')