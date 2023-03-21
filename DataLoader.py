# coding: utf-8
from functools import cmp_to_key

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from torch.utils.data import Dataset
from torchvision.io import read_image
import albumentations as A


# сортируем изображения по разрешению (чтобы в битче были изображения похожего разрешения)
def cmp_by_image_resolution(x, y):
    if x[2] > y[2]:
        return -1
    elif x[2] == y[2]:
        if x[3] > y[3]:
            return -1
        else:
            return 1
    else:
        return 1


class CarBodyDataset(Dataset):
    def __init__(self, img_paths, mask_paths, transform: A = None):
        assert len(img_paths) == len(mask_paths), \
            f'Кол-во картинок и масок к ним не совпадает {(len(img_paths), len(mask_paths))}'

        # сортируем изображения по разрешению (чтобы в битче были изображения похожего разрешения)
        img_with_size = []
        for i, path in enumerate(img_paths):
            chanels, height, width = read_image(str(path)).shape
            img_with_size.append((i, path, height, width))

        img_with_size.sort(key=cmp_to_key(cmp_by_image_resolution), reverse=False)
        new_sort_idx = [x[0] for x in img_with_size]

        self.len_dataset = len(img_paths)
        self.img_paths = np.array(img_paths)[new_sort_idx]
        self.mask_paths = np.array(mask_paths)[new_sort_idx]
        self._transform = transform

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, idx):
        img = cv2.imread(str(self.img_paths[idx]))

        mask = cv2.imread(str(self.mask_paths[idx]), cv2.IMREAD_GRAYSCALE)
        # 1 - это маска кузова авто
        mask_body = mask == 1

        if self._transform:
            augmented = self._transform(image=np.array(img, dtype=np.uint8), mask=np.array(mask_body, dtype=np.uint8))
            img = augmented['image']
            img = torch.FloatTensor(img)

            mask_body = augmented['mask']
            mask_body = torch.FloatTensor(mask_body)
        sample = {'img': img, 'mask': mask_body}
        return sample


def train_test_split(img_list: list, mask_list: list, size: float, permute=False) -> (list, list, list, list):
    """
    return: train_img_list, test_img_list, train_mask_list, test_mask_list
    """
    assert len(img_list) == len(mask_list), \
        f'Кол-во картинок и масок к ним не совпадает {(len(img_list), len(mask_list))}'
    assert 0 <= size <= 1, 'параметр size от 0 до 1'

    train_size = round(len(img_list) * size)
    if permute:
        idx = np.random.permutation(len(img_list))
    else:
        idx = list(range(len(img_list)))
    img_list_narray = np.array(img_list)
    permute_img_list = img_list_narray[idx]

    mask_list_narray = np.array(mask_list)
    permute_mask_list = mask_list_narray[idx]

    return permute_img_list[:train_size], permute_img_list[train_size:], \
           permute_mask_list[:train_size], permute_mask_list[train_size:]


class ToFloatTensor(object):
    def __call__(self, img):
        return img.type('torch.FloatTensor')
