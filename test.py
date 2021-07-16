import argparse
import os

import PIL
import cv2
import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from pytorchcv.models.unet import UNet
from torch import from_numpy
from torchvision import transforms
from torchvision.utils import save_image

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str, default="", help='Pfad (Ordner) für Testdaten')
parser.add_argument('--out_path', type=str, default="", help='Pfad (Ordner), wo Ausgabedaten hinsollen - muss nicht existieren')
parser.add_argument('--model_path', type=str, default="", help='Pfad (Datei), vom pretrained model')
arg = parser.parse_args()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Set Device

# define Input and Output Path
input_path = arg.input_path
out_path = arg.out_path
if not os.path.exists(out_path):
    os.makedirs(out_path)

# Create network, load weights
model_path = arg.model_path
'Example: model = ptcv_get_model("resnet18", pretrained=True)'
model = UNet(channels=[[128, 256, 512, 512], [512, 256, 128, 64]], init_block_channels=64, in_channels=3, num_classes=3, in_size=(128,128)).to(device=device)

model.load_state_dict(torch.load(model_path))
model.eval()
tf = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])


for filename in os.listdir(input_path):
    img = Image.open(os.path.join(input_path, filename))
    img = tf(img)

    img_tensor = torch.unsqueeze(img, 0)

    with torch.no_grad():
        img_out = model(img_tensor.cuda())

    pred = img_out[0].cpu().detach()
    pred = pred + img
    save_image(pred, out_path + filename)
