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

def align_image_to_16(img):
    """將圖片尺寸對齊到 16 的倍數 (支援 L 和 RGB 模式)"""
    w, h = img.size
    new_w = (w + 15) // 16 * 16
    new_h = (h + 15) // 16 * 16
    
    fill_color = "white" if img.mode == "L" else "black"
    padded_img = Image.new(img.mode, (new_w, new_h), fill_color)
    padded_img.paste(img, (0, 0))
    return padded_img


def save_figure_to_file(fig, filepath, mode='L'):
    """儲存 matplotlib figure 到檔案，並自動 16 對齊"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    
    buf.seek(0)
    img = Image.open(buf).convert(mode)
    img_aligned = align_image_to_16(img)  # 自動對齊到 16 的倍數
    img_aligned.save(filepath)


def setup_clean_axes(ax):
    """設定乾淨的座標軸（無刻度、無邊框）"""
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.margins(x=0, y=0)

# ===================== ECG 設定 =====================

class ECGPlotConfig:
    SEC_PER_INCH = 0.5
    HEIGHT_PER_LEAD = 0.6
    SEC_PER_INCH_COMPACT = 0.5
    HEIGHT_PER_LEAD_COMPACT = 0.5
    LINE_WIDTH = 0.8
    LINE_COLOR = 'black'


# ===================== ECG 波形函數 =====================

def wf_draw_single_lead(time, signal, figsize, dpi=64):
    """繪製單一 lead 的 ECG 波形"""
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111)
    
    ax.plot(time, signal, color=ECGPlotConfig.LINE_COLOR, linewidth=ECGPlotConfig.LINE_WIDTH)
    setup_clean_axes(ax)
    plt.tight_layout(pad=0)
    
    return fig


def wf_draw_combined_leads(time, ecg_data, dpi=64):
    """繪製多個 leads 在同一張圖"""
    n_leads = ecg_data.shape[1]
    fig_width = max(3, time[-1] * ECGPlotConfig.SEC_PER_INCH)
    fig_height = n_leads * ECGPlotConfig.HEIGHT_PER_LEAD
    
    fig, axes = plt.subplots(n_leads, 1, figsize=(fig_width, fig_height), sharex=True, dpi=dpi)
    axes = axes if n_leads > 1 else [axes]
    
    for i, ax in enumerate(axes):
        ax.plot(time, ecg_data[:, i], color=ECGPlotConfig.LINE_COLOR, linewidth=ECGPlotConfig.LINE_WIDTH)
        setup_clean_axes(ax)
    
    plt.subplots_adjust(hspace=0)
    plt.tight_layout(pad=0)
    
    return fig


def wf_draw_image(time, ecg_data, path, mode="combined", lead_names=None, height_per_lead=None, sec_per_inch=None, dpi=64):
    """生成 ECG 波形圖並儲存"""
    n_leads = ecg_data.shape[1]
    
    if mode == "combined":
        fig = wf_draw_combined_leads(time, ecg_data, dpi=dpi)
        save_figure_to_file(fig, path)
        return path
    else:
        paths = []
        figsize = (max(3, time[-1] * sec_per_inch), height_per_lead)
        
        for i in range(n_leads):
            fig = wf_draw_single_lead(time, ecg_data[:, i], figsize, dpi=dpi)
            lead_name = lead_names[i] if lead_names else f"lead{i}"
            file_path = Path(path).parent / f"{Path(path).stem}_{lead_name}.png"
            save_figure_to_file(fig, file_path)
            paths.append(file_path)
        
        return paths


# ===================== CWT 函數 =====================

def cwt_compute(signal, scales=None, wavelet='mexh'):
    """計算單一訊號的連續小波轉換"""
    if scales is None:
        scales = np.arange(1, 128)
    coef, _ = pywt.cwt(signal, scales, wavelet)
    return coef


def cwt_draw_single_image(coef, save_path, dpi=64, cmap='turbo'):
    """儲存單一 CWT 圖片"""
    H, W = coef.shape
    ratio = H / W
    fig_w = 5
    fig_h = fig_w * ratio
    
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    ax = fig.add_subplot(111)
    ax.imshow(coef, aspect='equal', origin='lower', cmap=cmap)
    ax.axis('off')
    plt.tight_layout(pad=0)
    
    save_figure_to_file(fig, save_path, mode='RGB')


def cwt_draw_combined_image(coef_list, save_path, dpi=64, cmap='turbo'):
    """儲存多個 leads 合併的 CWT 圖"""
    n_leads = len(coef_list)
    H, W = coef_list[0].shape
    ratio = H / W
    fig_w = 5
    fig_h = fig_w * ratio * n_leads
    
    fig, axes = plt.subplots(n_leads, 1, figsize=(fig_w, fig_h), dpi=dpi)
    axes = axes if n_leads > 1 else [axes]
    
    for ax, coef in zip(axes, coef_list):
        ax.imshow(coef, aspect='equal', origin='lower', cmap=cmap)
        ax.axis('off')
    
    plt.subplots_adjust(hspace=0)
    plt.tight_layout(pad=0)
    save_figure_to_file(fig, save_path, mode='RGB')


def cwt_draw_images(ecg_data, lead_names, path, mode="combined", dpi=64, cmap='turbo'):
    """計算並儲存 CWT 圖片"""
    scales = np.arange(1, 128)
    
    coef_list = [cwt_compute(ecg_data[:, i], scales) for i in range(ecg_data.shape[1])]
    
    if mode == "combined":
        # 直接使用傳入的 path，檔名加上 _cwt
        cwt_path = Path(path).parent / f"{Path(path).stem}_cwt.png"
        cwt_draw_combined_image(coef_list, cwt_path, dpi=dpi, cmap=cmap)
        return cwt_path
    else:
        # 和 wf_draw_image 一樣的邏輯
        paths = []
        for i, coef in enumerate(coef_list):
            lead_name = lead_names[i] if lead_names else f"lead{i}"
            file_path = Path(path).parent / f"{Path(path).stem}_cwt_{lead_name}.png"
            cwt_draw_single_image(coef, file_path, dpi=dpi, cmap=cmap)
            paths.append(file_path)
        return paths


# ===================== ECG 讀取與處理 =====================

def ecg_load_and_clean(root_dir, filename):
    """讀取 ECG 並使用 NeuroKit2 清洗"""
    record = wfdb.rdrecord(f"{root_dir}/{filename}")
    ecg = record.p_signal
    leads = record.sig_name
    fs = record.fs
    
    ecg_cleaned = np.array([nk.ecg_clean(ecg[:, i], sampling_rate=fs) for i in range(ecg.shape[1])]).T
    time = np.arange(ecg_cleaned.shape[0]) / fs
    
    return ecg_cleaned, time, leads, fs


def ecg_plot(root_dir, filename, mode="combined", use_cwt=False, dpi=64):
    """完整 ECG 波形與 CWT 圖生成流程"""
    out_dir = Path("vit_ecg_images") / filename
    out_dir.mkdir(parents=True, exist_ok=True)
    
    ecg_cleaned, time, leads, fs = ecg_load_and_clean(root_dir, filename)
    
    stem = Path(filename).stem
    if mode == "combined":
        final_path = out_dir / f"{stem}_12lead_vit.png"
    else:
        final_path = out_dir / f"{stem}.png"
        
    if use_cwt:
        cwt_draw_images(ecg_cleaned, leads, final_path, mode=mode, dpi=dpi)
    else:
        wf_draw_image(time, ecg_cleaned, final_path, mode=mode, lead_names=leads, dpi=dpi)

    
    return final_path


def ecg_process_record(task, mode="combined", use_cwt=False):
    """處理單筆 ECG 檔案"""
    path, filename = task
    try:
        ecg_plot(path, filename, mode, use_cwt=use_cwt)
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
        futures = [executor.submit(ecg_process_record, task, mode, use_cwt=False) for task in tasks]
        for future in as_completed(futures):
            print(future.result())
