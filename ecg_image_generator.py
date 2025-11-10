import wfdb
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

def plot_ecg_large(path: str, filename: str):
    out_dir = Path("vit_ecg_images") / filename
    """
    產生單一大圖（12導聯疊在一張圖）
    """
    # 讀取 ECG 訊號
    record = wfdb.rdrecord(f"{path}/{filename}")
    ecg = record.p_signal
    leads = record.sig_name
    fs = record.fs
    time = np.arange(ecg.shape[0]) / fs

    # 建立輸出資料夾
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    # 畫圖
    fig, axes = plt.subplots(len(leads), 1, figsize=(3.5, 3.5), sharex=True, dpi=64)
    for i, lead in enumerate(leads):
        ax = axes[i]
        ax.plot(time, ecg[:, i], linewidth=0.8)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.margins(x=0, y=0)

    plt.subplots_adjust(hspace=0)
    plt.tight_layout(pad=0)

    save_path = out_dir / f"{Path(filename).stem}_12lead_vit.png"
    fig.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()


def plot_ecg_small(path: str, filename: str):
    out_dir = Path("vit_ecg_images") / filename
    
    # 讀取 ECG 訊號
    record = wfdb.rdrecord(f"{path}/{filename}")
    ecg = record.p_signal
    leads = record.sig_name
    fs = record.fs
    time = np.arange(ecg.shape[0]) / fs

    # 建立輸出資料夾
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    for i, lead in enumerate(leads):
        fig = plt.figure(figsize=(3.5, 3.5), dpi=64)
        ax = plt.gca()
        
        # 繪製波形
        ax.plot(time, ecg[:, i], linewidth=0.8)
        
        # 完全移除所有邊距和軸
        ax.set_position([0, 0, 1, 1])
        ax.axis('off')
        
        save_path = out_dir / f"{Path(filename).stem}_{lead}.png"
        plt.savefig(save_path, dpi=64, pad_inches=0)
        plt.close()


# 根資料夾
root_dir = Path("ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/records100/")

# 建立任務列表，每個元素是 (path, filename)
tasks = []
for patient_dir in root_dir.iterdir():
    if patient_dir.is_dir():
        for record_file in patient_dir.iterdir():
            if record_file.is_file():
                filename = record_file.stem
                tasks.append((patient_dir, filename))

def process_record(task):
    path, filename = task
    try:
        plot_ecg_large(path, filename)
        return f"✅ Finished {filename}"
    except Exception as e:
        return f"❌ Error {filename}: {e}"

if __name__ == "__main__":
    root_dir = Path("ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/records100/")
    tasks = []
    for patient_dir in root_dir.iterdir():
        if patient_dir.is_dir():
            for record_file in patient_dir.iterdir():
                if record_file.is_file():
                    filename = record_file.stem
                    tasks.append((patient_dir, filename))

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_record, task) for task in tasks]
        for future in as_completed(futures):
            print(future.result())
