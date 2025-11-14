import wfdb
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import neurokit2 as nk
from PIL import Image
import io
from scipy.signal import stft
from concurrent.futures import ProcessPoolExecutor, as_completed

def draw_ecg_image(time, ecg_data, out_path, mode="combined", lead_names=None, height_per_lead=0.6,
                    sec_per_inch=0.5, dpi=64):

    def plot_and_save(fig, filename):
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf).convert("L")
        w, h = img.size
        new_w = (w + 15) // 16 * 16
        new_h = (h + 15) // 16 * 16
        padded_img = Image.new("L", (new_w, new_h), "white")
        padded_img.paste(img, (0, 0))
        padded_img.save(filename)

    n_leads = ecg_data.shape[1]

    # 所有 lead 同圖
    if mode == "combined":
        fig, axes = plt.subplots(n_leads, 1, figsize=(max(3, time[-1]*sec_per_inch), n_leads*height_per_lead),
                                 sharex=True, dpi=dpi)
        axes = axes if n_leads > 1 else [axes]
        for i, ax in enumerate(axes):
            ax.plot(time, ecg_data[:, i], color='black', linewidth=0.8)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values(): s.set_visible(False)
            ax.margins(x=0, y=0)
        plt.subplots_adjust(hspace=0); plt.tight_layout(pad=0)
        plot_and_save(fig, out_path)
        return out_path

    # 每條 lead 單獨圖
    else:
        paths = []
        for i in range(n_leads):
            fig = plt.figure(figsize=(max(3, time[-1]*sec_per_inch), height_per_lead), dpi=dpi)
            ax = fig.add_subplot(111)
            ax.plot(time, ecg_data[:, i], color='black', linewidth=0.8)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values(): s.set_visible(False)
            ax.margins(x=0, y=0)
            plt.tight_layout(pad=0)
            lead_name = lead_names[i] if lead_names else f"lead{i}"
            path = Path(out_path).parent / f"{Path(out_path).stem}_{lead_name}.png"
            plot_and_save(fig, path)
            paths.append(path)
        return paths


def plot_ecg(path: str, filename: str, mode, height_per_lead=0.6, sec_per_inch=0.5, dpi=64):
    out_dir = Path("vit_ecg_images") / filename
    out_dir.mkdir(parents=True, exist_ok=True)

    # 讀取 ECG
    record = wfdb.rdrecord(f"{path}/{filename}")
    ecg = record.p_signal
    leads = record.sig_name
    fs = record.fs

    # 清洗
    ecg_cleaned = np.array([nk.ecg_clean(ecg[:, i], sampling_rate=fs) 
                            for i in range(ecg.shape[1])]).T
    time = np.arange(ecg_cleaned.shape[0]) / fs

    if mode == "combined":
        final_path = out_dir / f"{Path(filename).stem}_12lead_vit.png"
    else:
        final_path = out_dir / f"{Path(filename).stem}.png"

    return draw_ecg_image(time, ecg_cleaned, final_path, mode=mode, lead_names=leads,
                          height_per_lead=height_per_lead, sec_per_inch=sec_per_inch,
                          dpi=dpi)

def process_record(task, mode):
    path, filename = task
    try:
        plot_ecg(path, filename, mode)
        return f"✅ Finished {filename}"
    except Exception as e:
        return f"❌ Error {filename}: {e}"

if __name__ == "__main__":
    root_dir = Path("ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/records100/")
    mode = "combined"
    tasks = []
    for patient_dir in root_dir.iterdir():
        if patient_dir.is_dir():
            for record_file in patient_dir.iterdir():
                if record_file.is_file():
                    filename = record_file.stem
                    tasks.append((patient_dir, filename))

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_record, task, mode) for task in tasks]
        for future in as_completed(futures):
            print(future.result())
