import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import torch.nn.functional as F
import timm
import os
from timm.data import create_transform
from pathlib import Path
from ecg_tool import load_model, predict_img

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load model
model = load_model(
    model_path='ecg_models/deit3_small_patch16_384_16_v5.pth',
    model_name='deit3_small_patch16_384.fb_in1k',
    device=device
)
model.eval()

data_config = timm.data.resolve_model_data_config(model.backbone)

test_transform = create_transform(
    **data_config,
    is_training=False
)

img_path = 'vit_ecg_images/hr/cwt/00002_hr_12lead_vit_cwt.png'

class_probs = predict_img(
    model=model,
    device=device,
    test_transform=test_transform,
    img_path=img_path,
)

print(class_probs)

# # ViT 專用 reshape_transform
# def reshape_transform(tensor, height=24, width=24):
#     result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
#     return result.transpose(2, 3).transpose(1, 2)

# # transform
# transform = T.Compose([
#     T.Resize((384, 384)),
#     T.ToTensor()
# ])

# # load image
# img = Image.open('vit_ecg_images/hr/cwt/00002_hr_12lead_vit_cwt.png').convert('RGB')
# tensor_img = transform(img).unsqueeze(0).to(device)

# # forward
# with torch.no_grad():
#     logits = model(tensor_img)
# probs = F.sigmoid(logits) 
# mask = probs[0] > 0.5
# indices = torch.nonzero(mask).squeeze(1)
# print(indices)

# save_folder = "vit_ecg_images/test/00002/"

# for class_idx in indices:
#     # GradCAM
#     cam = GradCAM(
#         model=model,
#         target_layers=[model.backbone.blocks[-1].norm1],
#         reshape_transform=reshape_transform
#     )

#     grayscale_cam = cam(
#         input_tensor=tensor_img,
#         targets=[ClassifierOutputTarget(class_idx)]
#     )[0]  # (384, 384)

#     # overlay
#     img_np = np.array(img.resize((384, 384))).astype(np.float32) / 255.0
#     cam_image = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
#     os.makedirs(save_folder, exist_ok=True)

#     save_path = os.path.join(save_folder, f"cam_class_{class_idx}.png")
#     img = Image.fromarray(cam_image)
#     img.save(save_path)