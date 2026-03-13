import torch
import random
from PIL import Image
import torchvision.transforms.functional as TF
import glob


img_paths = glob.glob('vit_ecg_images/hr/cwt/*.png')
print(len(img_paths))

print(0.1*len(img_paths))
