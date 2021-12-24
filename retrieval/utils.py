import os
import numpy as np
import time
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image

def data_path_loader(base_path, data_path, column_exist=True):
    output = []
    with open(data_path, 'r') as f:
        for idx, line in enumerate(f.readlines()):
            if idx==0 and column_exist:
                continue
            line = line.replace('\n', '')
            img_path, label_s, label_g, label_a = line.split(',')
            img_path = os.path.join(base_path, img_path)
            output.append([img_path, label_s, label_g, label_a])
    return output

def readable_path(img_path):
    _list = []
    for i in range(len(img_path)):
        try:
            Image.open(img_path[i][0])
            _list.append(True)
        except:
            _list.append(False)
    img_path = np.array(img_path)[_list, :].tolist()
    return img_path


class ArtPaintingDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_path_list = img_dir
        self.transform = transform
        # self.target_transform = target_transform
    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, idx):
        image_path = self.img_path_list[idx][0]
        image = read_image(image_path)
        label_dict = {}
        label = self.img_path_list[idx][1:]
        label_dict['style'] = label[0]
        label_dict['genre'] = label[1]
        label_dict['artist'] = label[2]
        if self.transform:
            image = self.transform(image)
        return image, label_dict, image_path

