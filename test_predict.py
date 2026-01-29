import torch
import torch.nn as nn
import timm
from ecg_eval import predict_ecg
from pathlib import Path

if __name__ == "__main__":
    # 1. 設定設備
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用設備: {device}")

    # 2. 載入模型權重與標籤名稱
    # 請確保此路徑下有模型檔案
    checkpoint_path = "ecg_models/vit_small_plus_dinov3_30.pth"
    
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"找不到模型檔案 {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    label_names = checkpoint['label_names']
    num_classes = len(label_names)

    # 3. 初始化模型結構
    model = timm.create_model('vit_small_plus_patch16_dinov3.lvd1689m', pretrained=True, num_classes=0)
    num_features = model.num_features
    model.head = nn.Linear(num_features, num_classes)
    
    # 載入權重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print("模型載入成功！")

    # 4. 準備測試影像路徑
    # 這裡選擇第一筆資料作為測試
    test_img_path = "vit_ecg_images/00001_lr/00001_lr_12lead_vit_cwt.png"
    
    if not Path(test_img_path).exists():
        raise FileNotFoundError(f"找不到測試影像 {test_img_path}")

    # 5. 呼叫 predict_ecg 進行預測
    pred_labels, pred_probs = predict_ecg(test_img_path, model, label_names, device, threshold=0.5)

    # 6. 顯示結果
    print(f"預測標籤: {pred_labels}")
    print(f"預測機率: {pred_probs}")
