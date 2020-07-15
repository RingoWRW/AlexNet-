"""
猫狗Dataset的构建
@author： 小吴
"""

import numpy as np
import os
from PIL import Image
import torch
import random
from torch.utils.data import Dataset

class CDDataset(Dataset):
    def __init__(self, data_dir, mode="train", split_n=0.9, rng_seed=620, transform=None):
        """
        rmb面额分类任务的Dataset
        :param data_dir:  数据集所在路径
        :param mode: train 和 valid可选
        :param rng_seed: 随机数
        :param transform: ，数据预处理
        """
        self.mode = mode
        self.data_dir = data_dir
        self.rng_seed = rng_seed
        self.split_n = split_n
        self.data_info = self._get_img_info()  # data_info存储所有图片路径和标签，在DataLoader中通过index读取样本
        self.transform = transform

    def __getitem__(self, index):
        path_img, label = self.data_info[index]
        img = Image.open(path_img).convert('RGB')     # 0~255

        if self.transform is not None:
            img = self.transform(img)   # 在这里做transform，转为tensor等等

        return img, label

    def __len__(self):
        if len(self.data_info) == 0:
            raise Exception("\ndata_dir:{} is a empty dir! Please checkout your path to images!".format(self.data_dir))
        return len(self.data_info)

    def _get_img_info(self):
        """
        attention 路径是否正确
        :return: 数据集(路径，标签)
        """
        img_names = os.listdir(self.data_dir)
        img_names = list(filter(lambda x: x.endswith('.jpg'), img_names))
        #保证每次epoch的数据都一样
        random.seed(self.rng_seed)
        random.shuffle(img_names)
        #cat->0 dog->1
        img_labels = [0 if n.startswith('cat') else 1 for n in img_names]
        split_idx = int(len(img_labels) * self.split_n)

        if self.mode == "train":
            img_set = img_names[:split_idx]
            label_set = img_labels[:split_idx]
        elif self.mode == "valid":
            img_set = img_names[split_idx:]
            label_set = img_labels[split_idx:]
        else:
            raise Exception("self.mode 无法识别，仅支持(train, valid)")

        path_img_set = [os.path.join(self.data_dir, n) for n in img_set]
        data_info = [(n, l) for n, l in zip(path_img_set, label_set)]

        return data_info