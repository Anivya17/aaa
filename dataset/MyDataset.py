from torch.utils.data import Dataset
from PIL import Image
import os
import torch.utils.data as data

#
from torchvision import transforms

class MyDataset(Dataset):
    def __init__(self, data_path, name):
        #paths
        if "psm" in name:
            self.input_path = data_path + "shadow_input/"
            self.gt_path = data_path + "shadow_gt/"
            self.portrait_mask = data_path + "shadow_input_mask_ind/"
        if name=="istd":
            self.input_path = data_path + "train_A/"
            self.gt_path = data_path + "train_C/"
            self.portrait_mask = None
        # self.segmentation = data_path + "portrait_segmentation/"
        # self.shadow_path = data_path + "shadow/"

        #transforms
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        self.gt_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()])

    def __getitem__(self, idx):
        # open input + gt
        # zusammengehörende Dateien müssen den gleichen Namen haben
        img_name = os.listdir(self.input_path)[idx]
        img = Image.open(os.path.join(self.input_path, img_name))
        gt = Image.open(os.path.join(self.gt_path, img_name))

        # Transformation ausführen
        img = self.transform(img)
        gt = self.gt_transform(gt)

        if not self.portrait_mask is None:
            portrait_mask = Image.open(os.path.join(self.portrait_mask, img_name))
            portrait_mask = self.gt_transform(portrait_mask)
            return img, gt, portrait_mask
        return img, gt, None

    def __len__(self):
        return len(os.listdir(self.input_path))

def get_loader(arg, batchsize):
    dataset = MyDataset(arg.path, arg.dataset_name)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=True)
    return data_loader

def get_validation_loader(arg, batchsize):
    validation_set = MyDataset(os.path.join(arg.path + "validation/"), arg.dataset_name)
    validation_loader = data.DataLoader(dataset=validation_set)
    return validation_loader
