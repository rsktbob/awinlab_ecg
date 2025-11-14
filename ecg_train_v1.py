import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import timm
from pathlib import Path
import ast
from sklearn.model_selection import train_test_split
import numpy as np
import wfdb
import neurokit2 as nk
from scipy.signal import stft

# Dataset with STFT preprocessing
class ECGSTFTDataset(Dataset):
    def __init__(self, data_paths, labels=None, nperseg=256, noverlap=128):
        self.data_paths = data_paths
        self.labels = labels
        self.nperseg = nperseg
        self.noverlap = noverlap
        
    def __len__(self):
        return len(self.data_paths)
    
    def ecg_to_stft(self, signal, fs):
        """Convert ECG signal to STFT representation"""
        stft_list = []
        for i in range(signal.shape[1]):
            f, t, Zxx = stft(signal[:, i], fs=fs, nperseg=self.nperseg, noverlap=self.noverlap)
            stft_list.append(np.abs(Zxx))  # Use magnitude
        stft_array = np.stack(stft_list, axis=0)  # [n_leads, freq_bins, time_bins]
        return stft_array
    
    def __getitem__(self, index):
        data_path = self.data_paths[index]
        
        # Load and clean ECG signal
        record = wfdb.rdrecord(str(data_path))
        signals = record.p_signal
        signals_cleaned = np.array([
            nk.ecg_clean(signals[:, i], sampling_rate=record.fs, method='neurokit') 
            for i in range(signals.shape[1])
        ]).T
        
        # Convert to STFT
        stft_features = self.ecg_to_stft(signals_cleaned, record.fs)
        
        # Normalize
        stft_features = (stft_features - stft_features.mean()) / (stft_features.std() + 1e-8)
        stft_features = torch.tensor(stft_features, dtype=torch.float32)
        
        if self.labels is not None:
            return stft_features, torch.tensor(self.labels[index], dtype=torch.float32)
        return stft_features


# Parse labels
def parse_scp_codes_with_probs(s):
    try:
        d = ast.literal_eval(s)
        d = {k: v for k, v in d.items() if v > 0}
        return d if d else {'NORM': 100.0}
    except:
        return {'NORM': 100.0}


if __name__ == "__main__":
    # Load labels
    df = pd.read_csv('ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/ptbxl_database.csv')
    all_dicts = [parse_scp_codes_with_probs(s) for s in df['scp_codes']]
    all_names = sorted({k for d in all_dicts for k in d})
    labels_prob = np.array([
        [d.get(name, 0)/100.0 for name in all_names] for d in all_dicts
    ], dtype=np.float32)
    
    print(f"Number of classes: {len(all_names)}")
    print(f"Total samples: {len(df)}")
    
    # Get data paths
    root_dir = Path("ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3")
    data_paths = []
    for idx, row in df.iterrows():
        filename_path = row['filename_lr']  # or 'filename_hr' for high resolution
        full_path = root_dir / filename_path
        data_paths.append(str(full_path).replace('.hea', ''))
    
    # Train/test split
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        data_paths, labels_prob, test_size=0.1, random_state=42
    )
    
    print(f"Train samples: {len(train_paths)}")
    print(f"Test samples: {len(test_paths)}")
    
    # Create datasets
    train_dataset = ECGSTFTDataset(train_paths, labels=train_labels, nperseg=256, noverlap=128)
    test_dataset = ECGSTFTDataset(test_paths, labels=test_labels, nperseg=256, noverlap=128)
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)
    
    # Get sample to determine input dimensions
    sample_data, _ = train_dataset[0]
    n_leads, freq_bins, time_bins = sample_data.shape
    print(f"STFT shape: [{n_leads}, {freq_bins}, {time_bins}]")
    
    # Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create ViT model with custom input channels (12 leads as channels)
    model = timm.create_model(
        'vit_small_patch16_224', 
        pretrained=False,
        in_chans=n_leads,  # 12 ECG leads as input channels
        img_size=(freq_bins, time_bins),  # STFT dimensions
        num_classes=len(all_names)
    )
    model.to(device)
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    
    torch.backends.cudnn.benchmark = True
    
    # Training loop
    num_epochs = 20
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        running_loss = 0.0
        
        for batch_idx, (stft_data, labels) in enumerate(train_loader):
            stft_data, labels = stft_data.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(stft_data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * stft_data.size(0)
            
            if (batch_idx + 1) % 50 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx+1}/{len(train_loader)}] - Loss: {loss.item():.4f}")
        
        avg_train_loss = running_loss / len(train_dataset)
        scheduler.step()
        
        # Validation
        model.eval()
        test_loss = 0.0
        all_preds, all_labels_list = [], []
        
        with torch.no_grad():
            for stft_data, labels in test_loader:
                stft_data, labels = stft_data.to(device), labels.to(device)
                outputs = model(stft_data)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * stft_data.size(0)
                
                preds = torch.sigmoid(outputs) > 0.5
                all_preds.append(preds.cpu().numpy())
                all_labels_list.append(labels.cpu().numpy())
        
        test_loss /= len(test_dataset)
        all_preds = np.vstack(all_preds)
        all_labels_list = np.vstack(all_labels_list)
        accuracy = (all_preds == all_labels_list).mean()
        
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}\n")
        
        # Save best model
        if test_loss < best_loss:
            best_loss = test_loss
            save_dir = Path("ecg_models")
            save_dir.mkdir(exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'test_loss': test_loss,
                'test_accuracy': accuracy,
                'label_names': all_names,
                'stft_params': {
                    'nperseg': 256,
                    'noverlap': 128,
                    'input_shape': (n_leads, freq_bins, time_bins)
                }
            }, save_dir / 'ecg_stft_vit_best.pth')
            print(f"✓ Best model saved (test_loss: {test_loss:.4f})")
    
    print("\nTraining completed!")
    print(f"Best test loss: {best_loss:.4f}")
    

    