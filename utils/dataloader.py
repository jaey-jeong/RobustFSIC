import torch
from torch.utils.data import Dataset
from torchvision import transforms

import numpy as np

from utils.utils import load_pickle, key_to_class, value_to_class
#from pgd import projected_gradient_descent

class FSICdataloader(Dataset) :
    def __init__(self, data) :
        super(FSICdataloader).__init__()
        self.data = load_pickle(data)

        self.imgs = self.data['image_data']
        self.lables = value_to_class(key_to_class(self.data['class_dict']))
        assert len(self.imgs) == len(self.lables), "there is a difference between size of images and labels"
        
        self.mean_pix = [x/255.0 for x in [120.39586422,  115.59361427, 104.54012653]]
        self.std_pix = [x/255.0 for x in [70.68188272,  68.27635443,  72.54505529]]

        self.data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = self.mean_pix,
                                std = self.std_pix)
        ])

    def __getitem__(self, idx) :
        x = self.data_transforms(self.imgs[idx])
        y = self.lables[idx]

        return x, y

    def __len__(self) :
        return len(self.imgs)

    def class_size(self) :
        return len(self.data['class_dict'])