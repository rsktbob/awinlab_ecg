import wfdb
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import neurokit2 as nk
from PIL import Image
import io
from scipy.signal import stft
from concurrent.futures import ProcessPoolExecutor, as_completed

import io
from pathlib import Path

import numpy as np
import neurokit2 as nk
import wfdb
import pywt
import matplotlib.pyplot as plt
from PIL import Image


import io
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import wfdb
import neurokit2 as nk
import pywt


# ===================== 工具函數 =====================

def save_figure_to_file(fig, filepath):
    """儲存 matplotlib figure 到檔案"""
    plt.tight_layout(pad=0)
    fig.savefig(filepath, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def setup_clean_axes(ax):
    """設定乾淨的座標軸"""
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.margins(x=0, y=0)

# ===================== ECG 波形函數 =====================

def wf_draw_single_lead(time, signal, dpi=32):
    """繪製單一 lead 的 ECG 波形"""
    fig = plt.figure(figsize=(10, 2), dpi=dpi)
    ax = fig.add_subplot(111)
    
    ax.plot(time, signal, color='black', linewidth=0.8)
    setup_clean_axes(ax)
    
    return fig


def wf_draw_combined_leads(time, ecg_data, dpi=32):
    """繪製多個 leads 在同一張圖，不拉伸，消除縫隙"""
    n_leads = ecg_data.shape[1]
    
    fig, axes = plt.subplots(n_leads, 1, figsize=(10, 14), sharex=True, dpi=dpi)
    axes = axes if n_leads > 1 else [axes]
    
    fig.subplots_adjust(hspace=0)
    
    for i, ax in enumerate(axes):
        ax.plot(time, ecg_data[:, i], color='black', linewidth=0.8)
        setup_clean_axes(ax)
    
    return fig


def wf_draw_image(time, ecg_data, lead_names, filename, resolution, mode="combined", dpi=32):
    """生成 ECG 波形圖並儲存"""
    n_leads = ecg_data.shape[1]

    if mode == "combined":
        output_dir = Path("vit_ecg_images") / resolution / "wf"
        output_dir.mkdir(parents=True, exist_ok=True)
        filepath = output_dir / f"{filename}_12lead_vit.png"
        fig = wf_draw_combined_leads(time, ecg_data, dpi=dpi)
        save_figure_to_file(fig, filepath)
        return output_dir
    else:
        paths = []
        output_dir = Path("vit_ecg_images") / resolution / "wf" / Path(filename)
        output_dir.mkdir(parents=True, exist_ok=True)

        for i in range(n_leads):
            fig = wf_draw_single_lead(time, ecg_data[:, i], dpi=dpi)
            lead_name = lead_names[i] if lead_names else f"lead{i}"
            filepath = output_dir / f"{filename}_{lead_name}.png"
            
            save_figure_to_file(fig, filepath)
            paths.append(filepath)
        
        return paths


# ===================== CWT 函數 =====================

def cwt_compute(signal, scales=None, wavelet='mexh'):
    """計算單一訊號的連續小波轉換"""
    if scales is None:
        scales = np.arange(1, 128)
    coef, _ = pywt.cwt(signal, scales, wavelet)
    return coef


def cwt_draw_single_image(coef, dpi=32, cmap='turbo'):
    """儲存單一 CWT 圖片"""    
    fig = plt.figure(figsize=(10, 2), dpi=dpi)
    ax = fig.add_subplot(111)
    ax.imshow(coef, aspect='auto', origin='lower', cmap=cmap)
    ax.axis('off')
    
    return fig
    save_figure_to_file(fig, save_path, mode='RGB')


def cwt_draw_combined_image(coef_list, dpi=32, cmap='turbo'):
    """儲存多個 leads 合併的 CWT 圖"""
    n_leads = len(coef_list)
    
    fig, axes = plt.subplots(n_leads, 1, figsize=(14, 12), dpi=dpi)
    axes = axes if n_leads > 1 else [axes]
    
    fig.subplots_adjust(hspace=0)

    for ax, coef in zip(axes, coef_list):
        ax.imshow(coef, aspect='auto', origin='lower', cmap=cmap)
        setup_clean_axes(ax)

    return fig



def cwt_draw_image(ecg_data, lead_names, filename, resolution, mode="combined", dpi=32, cmap='turbo'):
    """計算並儲存 CWT 圖片"""
    scales = np.arange(1, 128)
    
    coef_list = [cwt_compute(ecg_data[:, i], scales) for i in range(ecg_data.shape[1])]
    
    if mode == "combined":
        fig = cwt_draw_combined_image(coef_list, dpi=dpi, cmap=cmap)
        output_dir = Path("vit_ecg_images") / resolution / "cwt"
        # output_dir.mkdir(parents=True, exist_ok=True)
        filepath = output_dir / f"{filename}_12lead_vit_cwt.png"
        save_figure_to_file(fig, filepath)
        return filepath
    else:
        paths = []
        for i, coef in enumerate(coef_list):
            lead_name = lead_names[i] if lead_names else f"lead{i}"
            fig = cwt_draw_single_image(coef, dpi=dpi, cmap=cmap)
            output_dir = Path("vit_ecg_images") / resolution / "cwt" / Path(filename)
            output_dir.mkdir(parents=True, exist_ok=True)
            filepath = output_dir / f"{filename}_{lead_name}.png"
            save_figure_to_file(fig, filepath)
            paths.append(filepath)
        return paths


# ===================== ECG 讀取與處理 ============

def ecg_load_and_clean(root_dir, filename):
    """讀取 ECG 並使用 NeuroKit2 清洗"""
    record = wfdb.rdrecord(f"{root_dir}/{filename}")
    ecg = record.p_signal
    leads = record.sig_name
    fs = record.fs
    
    ecg_cleaned = np.array([nk.ecg_clean(ecg[:, i], sampling_rate=fs) for i in range(ecg.shape[1])]).T
    time = np.arange(ecg_cleaned.shape[0]) / fs
    
    return ecg_cleaned, time, leads, fs


def ecg_plot(root_dir, filename, resolution, mode="combined", use_cwt=False, dpi=32):
    """完整 ECG 波形與 CWT 圖生成流程"""
    ecg_cleaned, time, lead_names, fs = ecg_load_and_clean(root_dir, filename)
            
    if use_cwt:
        cwt_draw_image(ecg_cleaned, lead_names=lead_names, filename=filename, resolution=resolution, mode=mode, dpi=dpi)
    else:
        wf_draw_image(time, ecg_cleaned, lead_names=lead_names, filename=filename, resolution=resolution, mode=mode, dpi=dpi)

    
    return filename


def ecg_process(task, resolution, mode="combined", use_cwt=False):
    """處理單筆 ECG 檔案"""
    root_dir, filename = task
    try:
        ecg_plot(root_dir, filename, resolution, mode, use_cwt=use_cwt)
        return f"✅ Finished {filename}"
    except Exception as e:
        return f"❌ Error {filename}: {e}"


if __name__ == "__main__":
    resolution = "lr"
    root_dir = "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/"
    root_dir = Path(root_dir + "records100") if resolution == "lr" else Path(root_dir + "records500")

    mode = "combined"
    tasks = []
    for patient_dir in root_dir.iterdir():
        if patient_dir.is_dir():
            for record_file in patient_dir.iterdir():
                if record_file.is_file():
                    filename = record_file.stem
                    tasks.append((patient_dir, filename))

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(ecg_process, task, resolution, mode, use_cwt=True) for task in tasks]
        for future in as_completed(futures):
            print(future.result())
