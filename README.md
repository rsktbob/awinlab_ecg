# 使用 Vision Transformers (ViT) 進行 ECG 心電圖分類

本專案實作了一個端到端的流程，利用 Vision Transformers (ViT) 對心電圖 (ECG) 訊號進行分類。程式將 12 導程 (12-lead) 的 ECG 訊號轉換為連續小波轉換 (CWT) 的頻譜圖 (spectrograms)，並基於 PTB-XL 資料集訓練深度學習模型以預測診斷標籤。

## 🚀 功能特色

- **ECG 訊號處理**：使用 `neurokit2` 對 ECG 訊號進行清洗與標準化。
- **影像生成**：使用 `pywt` 將一維 (1-D) ECG 訊號轉換為二維 (2-D) 的時頻表示圖 (CWT 頻譜圖)。
- **深度學習模型**：利用 `timm` 函式庫中先進的 Vision Transformers (ViT) 進行影像分類。
- **多標籤分類**：支援單一 ECG 紀錄對應多個診斷標籤的分類任務。
- **評估指標**：計算準確率 (Accuracy)、F1-Score (Macro) 以及 AUROC (Macro)。

## 📂 專案結構

```
.
├── ecg_image_generator.py  # 處理原始 ECG 資料並生成 CWT 影像的程式
├── ecg_train.py            # 訓練 ViT 模型的程式
├── ecg_eval.py             # 評估已訓練模型的程式
├── test_predict.py         # 用來展示predict_ecg函式的使用
├── requirements.txt        # Python 相依套件清單
├── venv/                   # Python 虛擬環境（此資料夾容量較大，超過 GitHub 上限，可用 requirements.txt 重新建立環境）     
├── ecg_models/             # 儲存訓練好的模型（此資料夾容量較大，超過 GitHub 上限，可從 Google Drive 下載：https://drive.google.com/drive/folders/1tM3R6hCHCNamfD-4ZcnQmtnL1Aq2bfz2?usp=drive_link）
├── vit_ecg_images/         # 儲存生成的 ECG 影像（此資料夾容量較大，可用 ecg_image_generator.py 產生影像）
└── ptb-xl-.../             # (外部) PTB-XL 資料集目錄（此部分需自行下載，可至 https://physionet.org/content/ptb-xl/1.0.3/）
```

## 🛠️ 安裝說明

1.  **複製專案 (Clone)** (若適用)。
2.  **安裝相依套件**：
    請確保已安裝 Python。建議使用虛擬環境 (Virtual Environment)。

    ```bash
    pip install -r requirements.txt
    ```

## 📊 資料準備

本專案使用 **PTB-XL** 資料集。

1.  **下載資料集**：確保 PTB-XL 資料集位於專案根目錄下 `ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/`。(可到https://physionet.org/content/ptb-xl/1.0.3進行下載)
2.  **生成影像**：
    執行生成腳本將 ECG 訊號轉換為 CWT 影像。此過程使用多行程 (multiprocessing) 加速轉換。

    ```bash
    python ecg_image_generator.py
    ```
    *輸出*：影像將被儲存於 `vit_ecg_images/` 目錄中。

## 🧠 模型訓練

訓練 Vision Transformer 模型：

```bash
python ecg_train.py
```

- **模型架構**：`vit_small_patch16_dinov3` (透過 `timm` 載入)。
- **輸入資料**：來自 `vit_ecg_images/` 的 CWT 影像。
- **輸出結果**：最佳模型權重將儲存至 `ecg_models/ecg_vit_model_v8.pth`。

## 📉 模型評估

在測試集上評估模型效能：

```bash
python ecg_eval.py
```

此腳本將執行以下動作：
- 載入訓練好的模型。
- 計算 **Accuracy (準確率)**、**F1-Score** 和 **AUROC**。
- 對範例影像進行預測並輸出結果。

## 🧪 專案使用說明

ecg_eval.py 中的 predict_ecg(img_path, model, label_names, device, threshold=0.5) 函式會回傳機率高於設定 threshold 的標籤與對應機率。使用方式可以參考test_predict.py。