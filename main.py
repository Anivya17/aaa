#Imports
import argparse
import torch
from pytorchcv.model_provider import get_model as ptcv_get_model
from pytorchcv.models.sinet import SINet
from pytorchcv.models.unet import UNet
from torch.optim import lr_scheduler

from dataset.MyDataset import get_loader, get_validation_loader
from model.GridNet import GridNet
from model.SINet import get_sinet
from train import train

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=100, help='epoch number')
parser.add_argument('--path', type=str, default="", help='Pfad für Datensatz')
parser.add_argument('--dataset_name', type=str, default="", help='Name von Datensatz: istd, psm')
parser.add_argument('--model_name', type=str, default="", help='Name von Model: unet')
arg = parser.parse_args()

# Hyperparameters
learning_rate = 0.003
batchsize = 16
num_epochs = arg.epoch

# Load Data - hier Path einstellen
# ? save_path=""
path = arg.path
dataloader = get_loader(arg, batchsize)
validation_loader = get_validation_loader(arg, batchsize)

# Create network
'Example: model = ptcv_get_model("resnet18", pretrained=True)'
if arg.model_name == 'unet':
    model = UNet(channels=[[128, 256, 512, 512], [512, 256, 128, 64]], init_block_channels=64, in_channels=3, num_classes=3, in_size=(128,128))
if arg.model_name == 'gridnet':
    model = GridNet(in_chs=3, out_chs=3)
if arg.model_name == 'sinet':
    model = get_sinet()
if arg.model_name == 'pspnet':
    model = ptcv_get_model("pspnet_resnetd50b_voc", num_classes=3, in_size=(128,128), aux=False)
if arg.model_name == 'deeplabv3':
    model = ptcv_get_model("deeplabv3_resnetd50b_voc", num_classes=3)

# model.load_state_dict(torch.load("C:\\Users\\Fiona\\PycharmProjects\\aaa\\trained_models\\istd_workshard10.pth"))
# Loss and optimizer
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

# Train Network
train(dataloader, model, loss_fn, optimizer, scheduler, arg, validation_loader)
# train(dataloader, model, num_epochs, loss_fn, optimizer, scheduler, arg)

# check accuracy of network