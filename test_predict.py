import torch
import torch.nn as nn
import timm
from pathlib import Path
from ecg_tool import load_model, predict_img
from timm.data import create_transform


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用設備: {device}")
    model = load_model(model_path='ecg_models/deit3_small_patch16_384_16_v3.pth', 
                       model_name='deit3_small_patch16_384.fb_in1k',
                       device=device)
    
    model.eval()

    data_config = timm.data.resolve_model_data_config(model.backbone)

    test_transform = create_transform(
        **data_config,
        is_training=False
    )

    class_probs = predict_img(model=model, 
                              device=device, 
                              img_path='vit_ecg_images/hr/cwt/00002_hr_12lead_vit_cwt.png',
                              test_transform=test_transform,
                              is_attention_map=True)
    
    print(class_probs)
