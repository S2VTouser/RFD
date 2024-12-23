# -*- coding: utf-8 -*-

import os
import glob
from typing import List
import yaml

import torch
import torchvision as tv

from .base import Dataset, ImageMetaDataset, MetaItem


class ImageNetMetaDataset(ImageMetaDataset):
    _meta_files = dict(train="train.txt", val="val.txt")

    def setup_meta(self) -> List[MetaItem]:
        meta_file = self._meta_files["train" if self.train else "val"]

        with open(os.path.join(self.root, meta_file)) as f:
            lines = f.readlines()

        meta = []
        for line in lines:
            fpath, target = line[:-1].split(" ")
            fpath, target = os.path.join(self.root, fpath), int(target)
            #print('~~~', fpath, target)
            meta.append(MetaItem(fpath, target))
        return meta


class ImageNetMetaDataset100(ImageNetMetaDataset):
    _meta_files = dict(train="train_100.txt", val="val_100.txt")


class ImageNetMetaDataset1000(ImageNetMetaDataset):
    _meta_files = dict(train="train_1000.txt", val="val_1000.txt")


class ImageNetRDataset(ImageNetMetaDataset):

    def setup_meta(self) -> List[MetaItem]:

        if self.train:
            data_config = yaml.load(open('./data/imagenet-r_train.yaml', 'r'), Loader=yaml.Loader)
        else:
            data_config = yaml.load(open('./data/imagenet-r_test.yaml', 'r'), Loader=yaml.Loader)
        #print(data_config['targets'])
        # self.data = data_config['data']
        # self.targets = data_config['targets']
        #print(data_config['targets'])
        meta = []
        n = 0
        for i, j in zip(data_config['data'], data_config['targets']):
            target = int(j)
            #print('meta content:', i, target)
            meta.append(MetaItem(i, target))
        return meta


class ImageNet100(Dataset):
    dataset_cls = ImageNetMetaDataset100

    @property
    def transform(self):
        ts = [
            tv.transforms.Resize(256),
            tv.transforms.CenterCrop(224),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(
                (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
            ),
        ]
        return tv.transforms.Compose(ts)

    @property
    def dims(self):
        return torch.Size([3, 224, 224])


class ImageNet1000(ImageNet100):
    dataset_cls = ImageNetMetaDataset1000


class ImageNetR(Dataset):
    dataset_cls = ImageNetRDataset

    @property
    def transform(self):
        ts = [
            tv.transforms.Resize(256),
            tv.transforms.CenterCrop(224),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(
                (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
            ),
        ]
        return tv.transforms.Compose(ts)

    @property
    def dims(self):
        return torch.Size([3, 224, 224])
