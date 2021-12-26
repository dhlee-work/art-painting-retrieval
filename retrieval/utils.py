import os
from itertools import chain

import numpy as np
import time
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import matplotlib.pyplot as plt

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

def load_db(db_path='./data/DB/autoencoder'):
    data_db_train = np.load(os.path.join(db_path, 'result_train_inference_epoch50.npy'),
                            allow_pickle=True).item()
    data_db_test = np.load(os.path.join(db_path, 'result_test_inference_epoch50.npy'),
                           allow_pickle=True).item()
    data_db = {}
    for idx, key in enumerate(chain(data_db_train, data_db_test)):
        data_db[idx] = data_db_train[key]
    len(data_db)
    db_feature = np.zeros((len(data_db),
                           len(np.array(data_db_train[0]['feature']).reshape(-1).tolist()) + 1))
    print(f'd_db shape : {db_feature.shape}')
    for idx, key in enumerate(data_db):
        db_feature[idx, :] = np.array([key] + np.array(data_db[key]['feature']).reshape(-1).tolist())
    return data_db, db_feature


def vis_retrieval(detected_img, data_db, data_q_sim):
    plt.figure(figsize=(30, 15))
    dummy_i = 1
    i = 0
    intr_sim = data_q_sim[i, :]
    sort_val = np.argsort(1 - intr_sim)
    for _iter in range(11):
        rank = _iter
        if _iter == 0:
            plt.subplot(2, 6, rank + dummy_i)
            img = detected_img

            plt.imshow(img)
            plt.xlabel(f'query',  # \n size:{_size}
                       fontsize=15)
            plt.xticks([], [])
            plt.yticks([], [])

        if _iter == 6:
            dummy_i = + dummy_i + 1

        if _iter != 0:
            _sim = round(intr_sim[sort_val[rank - 1]], 3)
            _name = os.path.splitext(data_db[sort_val[rank - 1]]['name'])[0]
            _path = data_db[sort_val[rank - 1]]['path']

            plt.subplot(2, 6, rank + dummy_i)
            img = plt.imread(_path)
            _size = img.shape[:2]
            _size = ','.join([str(i) for i in _size])
            _name2 = _name[:len(_name) // 2] + '\n ' + _name[len(_name) // 2:]
            plt.imshow(img)
            plt.xlabel(f'{_name2}\n similarity:{_sim}',
                       fontsize=15)
            plt.xticks([], [])
            plt.yticks([], [])

        if _iter == 0:
            plt.title('Query')
        else:
            plt.title(f'Top {rank}')
    plt.show()
    plt.savefig(f'./result.jpg')
    plt.close()


