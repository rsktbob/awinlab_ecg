import torch
from torch import nn
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from torch.utils.data import DataLoader
import timm
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset
import pandas as pd
import ast
from ecg_train import ECGDataset

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
    model.head = nn.Linear(num_features, num_classes)
    
    # 載入權重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✅ 模型載入成功！")
    print(f"   訓練 epoch: {checkpoint['epoch']}")
    print(f"   訓練 loss: {checkpoint['train_loss']:.4f}")
    print(f"   測試 loss: {checkpoint['test_loss']:.4f}")
    print(f"   測試 accuracy: {checkpoint['test_accuracy']:.4f}")
    print(f"   類別: {class_names}\n")
    
    return model, class_names


# 準備測試數據（要和訓練時的處理一樣）
def prepare_test_data():
    # Load scp_statements.csv
    agg_df = pd.read_csv('ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/scp_statements.csv', index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]
    
    def aggregate_diagnostic_superclass(y_dic):
        tmp = {}
        for key, value in y_dic.items():
            if key in agg_df.index:
                if value > 0:
                    tmp[agg_df.loc[key].diagnostic_class] = value
        return tmp
    
    # Load database
    df = pd.read_csv('ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/ptbxl_database.csv', index_col='ecg_id')
    df.scp_codes = df.scp_codes.apply(lambda x: ast.literal_eval(x))
    df['diagnostic_superclass'] = df.scp_codes.apply(aggregate_diagnostic_superclass)
    df = df[df.diagnostic_superclass.map(lambda x: len(x)) > 0]
    
    class_names = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
    
    # Load image paths
    img_paths = []
    for path in df['filename_lr']:
        img_paths.append(Path("vit_ecg_images/lr/cwt") / f"{Path(path).stem}_12lead_vit_cwt.png")
    
    # Load labels
    labels_prob = np.array([
        [d.get(name, 0)/100.0 for name in class_names] 
        for d in df['diagnostic_superclass']
    ], dtype=np.float32)
    
    # 使用和訓練時相同的分割
    from sklearn.model_selection import train_test_split
    _, test_img_paths, _, test_labels = train_test_split(
        img_paths, labels_prob, test_size=0.1, random_state=42
    )
    
    print(f"✅ 測試數據準備完成！")
    print(f"   測試樣本數: {len(test_img_paths)}")
    print(f"   標籤 shape: {test_labels.shape}\n")
    
    return test_img_paths, test_labels, class_names


# 評估模型
def evaluate_model(model, test_loader, device, class_names):
    model.eval()
    
    predicted_probs = []
    predicted_binary_labels = []
    true_binary_labels = []
    
    print("開始評估...")
    with torch.no_grad():
        for batch_images, batch_labels in test_loader:
            batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)
            logits = model(batch_images)
            prob = torch.sigmoid(logits)
            
            predicted_probs.append(prob.cpu().numpy())
            predicted_binary_labels.append((prob > 0.5).cpu().numpy())
            true_binary_labels.append((batch_labels > 0.5).cpu().numpy())

    # 合併所有批次
    predicted_probs = np.vstack(predicted_probs)
    predicted_binary_labels = np.vstack(predicted_binary_labels)
    true_binary_labels = np.vstack(true_binary_labels)
    
    print(f"預測完成！")
    print(f"   預測 shape: {predicted_binary_labels.shape}")
    print(f"   標籤 shape: {true_binary_labels.shape}\n")
    
    # 計算指標
    print("="*60)
    print("整體指標")
    print("="*60)
    
    # 計算精準度
    accuracy = accuracy_score(true_binary_labels, predicted_binary_labels)
    
    # 計算 F1 Score
    f1_macro = f1_score(true_binary_labels, predicted_binary_labels, average='macro', zero_division=0)
    
    # 找出有效的標籤（避免只有一個類別的情況）
    valid_class_indices = []
    for i in range(true_binary_labels.shape[1]):
        if len(np.unique(true_binary_labels[:, i])) > 1:
            valid_class_indices.append(i)
    
    # 計算 AUROC
    if len(valid_class_indices) > 0:
        auroc_macro = roc_auc_score(
            true_binary_labels[:, valid_class_indices], 
            predicted_probs[:, valid_class_indices], 
            average='macro'
        )
    else:
        auroc_macro = 0.0
        print("警告: 所有類別都只有單一標籤，無法計算 AUROC")
    
    # 輸出結果
    print(f"Accuracy:          {accuracy:.4f}")
    print(f"F1-Score (Macro):  {f1_macro:.4f}")
    print(f"AUROC (Macro):     {auroc_macro:.4f}")
    
    # 每個類別的詳細報告
    print("\n" + "="*60)
    print("各類別詳細指標")
    print("="*60)
    print(classification_report(
        true_binary_labels, 
        predicted_binary_labels, 
        target_names=class_names,
        zero_division=0,
        digits=4
    ))
    
    # 每個類別的 AUROC
    print("="*60)
    print("各類別 AUROC")
    print("="*60)
    for i, class_name in enumerate(class_names):
        if len(np.unique(true_binary_labels[:, i])) > 1:
            class_auroc = roc_auc_score(true_binary_labels[:, i], predicted_probs[:, i])
            print(f"{class_name:6s}: {class_auroc:.4f}")
        else:
            print(f"{class_name:6s}: N/A (只有單一類別)")
    
    # 標籤分佈統計
    print("\n" + "="*60)
    print("標籤分佈統計")
    print("="*60)
    for i, class_name in enumerate(class_names):
        positive_samples = true_binary_labels[:, i].sum()
        total_samples = len(true_binary_labels)
        print(f"{class_name:6s}: {positive_samples:5.0f}/{total_samples} ({positive_samples/total_samples*100:5.2f}%)")
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'auroc_macro': auroc_macro,
        'predicted_probabilities': predicted_probs,
        'predicted_binary_labels': predicted_binary_labels,
        'true_binary_labels': true_binary_labels
    }


if __name__ == "__main__":
    # 設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = "ecg_models/vit_small_plus_dinov3_30_v4.pth"
    
    print(f"使用裝置: {device}\n")
    
    # 1. 載入模型
    model, class_names = load_model(checkpoint_path, device)
    
    # 2. 準備測試數據
    test_img_paths, test_labels, _ = prepare_test_data()
    
    # 3. 建立 DataLoader
    test_dataset = ECGDataset(test_img_paths, labels=test_labels)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, 
                            num_workers=0, pin_memory=True)
    
    # 4. 評估模型
    results = evaluate_model(model, test_loader, device, class_names)
    
    print("\n✅ 評估完成！")