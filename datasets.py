import os
import torch
import pandas as pd
from skimage import io
from skimage import color
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import warnings
import numpy as np
warnings.filterwarnings("ignore")

class ImageNette_train(Dataset):

    #mapping = {"n01440764":0, "n02102040":217, "n02979186":482, "n03000684":491, "n03028079":497, "n03394916":566, "n03417042":569, "n03425413":571, "n03445777":574, "n03888257":701}
    """
      0: tench
      217: English springer
      482: cassette player
      491: chain saw
      497: church
      566: French horn
      569: garbage truck
      571: gas pump
      574: golf ball
      701: parachute
    """
    mapping = {"n01440764": 0, "n02102040": 1, "n02979186": 2, "n03000684": 3, "n03028079": 4, "n03394916": 5,"n03417042": 6, "n03425413": 7, "n03445777": 8, "n03888257": 9}

    def __init__(self, root_dir=None, csv_file="noisy_imagenette.csv"):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.datas = pd.read_csv(os.path.join(root_dir, csv_file))
        self.root_dir = root_dir
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomRotation(30),
            transforms.RandomHorizontalFlip(),
            transforms.GaussianBlur(kernel_size=5),
            transforms.ColorJitter(),
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.datas.iloc[idx, 0])
        image = io.imread(img_name)
        if len(image.shape) != 3:
            image = color.gray2rgb(image)         
        label = self.datas.iloc[idx, 1]
        label = self.mapping[label]
        label = torch.tensor(label)
    
        image = self.transform(image.copy())

        sample = (image, label)

        return sample
class ImageNette_test(Dataset):

    #mapping = {"n01440764":0, "n02102040":217, "n02979186":482, "n03000684":491, "n03028079":497, "n03394916":566, "n03417042":569, "n03425413":571, "n03445777":574, "n03888257":701}
    mapping = {"n01440764": 0, "n02102040": 1, "n02979186": 2, "n03000684": 3, "n03028079": 4, "n03394916": 5, "n03417042": 6, "n03425413": 7, "n03445777": 8, "n03888257": 9}

    def __init__(self, root_dir=None, csv_file="test.csv"):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.datas = pd.read_csv(os.path.join(root_dir, csv_file))
        self.root_dir = root_dir
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.datas.iloc[idx, 0])
        image = io.imread(img_name)
        if len(image.shape) != 3:
            image = color.gray2rgb(image)
        label = self.datas.iloc[idx, 1]
        label = self.mapping[label]
        label = torch.tensor(label)

        image = self.transform(image.copy())

        sample = (image, label)

        return sample

class CUB_train(Dataset):
    def __init__(self, root_dir=None, csv_file="noisy_imagenette.csv"):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.datas = pd.read_csv(os.path.join(root_dir, csv_file))
        self.root_dir = root_dir
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((600, 600), Image.BILINEAR),
            transforms.RandomCrop((448, 448)),
            transforms.RandomHorizontalFlip(),
        ])

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.datas.iloc[idx, 1])
        image = io.imread(img_name)
        seg_name = os.path.join(self.root_dir, self.datas.iloc[idx, 3])
        seg = io.imread(seg_name)[:,:,None]
        label = self.datas.iloc[idx, 2]
        label = torch.tensor(label)
        image = np.concatenate([image, seg], axis=2)
        image = self.transform(image.copy())
        seg = image[3, :,:]
        image = image[:3,:,:]
        image = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image)
        sample = (image, seg, label)
        return sample

class CUB_test(Dataset):
    def __init__(self, root_dir=None, csv_file="noisy_imagenette.csv"):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.datas = pd.read_csv(os.path.join(root_dir, csv_file))
        self.root_dir = root_dir
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((600, 600), Image.BILINEAR),
            transforms.CenterCrop((448, 448)),
        ])

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.datas.iloc[idx, 1])
        image = io.imread(img_name)
        seg_name = os.path.join(self.root_dir, self.datas.iloc[idx, 3])
        seg = io.imread(seg_name)[:,:,None]
        label = self.datas.iloc[idx, 2]
        label = torch.tensor(label)
        image = np.concatenate([image, seg], axis=2)
        image = self.transform(image.copy())
        seg = image[3, :,:]
        image = image[:3,:,:]
        image = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image)
        sample = (image, seg, label)
        return sample
