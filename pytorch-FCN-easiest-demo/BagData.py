import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

import cv2
import numpy as np

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.
    Normalize(  # mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        mean=[0.588, 0.521, 0.496],
        std=[0.265, 0.268, 0.27])
])


def onehot(data, n):
    # buf = np.zeros(data.shape + (n, ))
    # nmsk = np.arange(data.size) * n + data.ravel()
    # buf.ravel()[nmsk] = 1
    mask0 = data + 1
    mask0[mask0 == 2] = 0
    return np.array([mask0, data])


class BagDataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform

    def __len__(self):
        return len(os.listdir('bag_data'))

    def __getitem__(self, idx):
        img_name = os.listdir('bag_data')[idx]
        imgA = cv2.imread('bag_data/' + img_name)
        imgA = cv2.resize(imgA, (160, 160))
        imgB = cv2.imread('bag_data_msk/' + img_name, 0)
        imgB = cv2.resize(imgB, (160, 160))
        # 原始目标mask像素值不只是[255,255,255],
        imgB[imgB > 220] = 255
        imgB = imgB / 255
        imgB = imgB.astype('uint8')
        imgB = onehot(imgB, 2)
        # imgB = imgB.transpose(2, 0, 1)
        imgB = torch.FloatTensor(imgB)
        # print(imgB.shape)
        if self.transform:
            imgA = self.transform(imgA)

        return imgA, imgB


bag = BagDataset(transform)

train_size = int(0.9 * len(bag))
test_size = len(bag) - train_size
train_dataset, test_dataset = random_split(bag, [train_size, test_size])

train_dataloader = DataLoader(
    train_dataset, batch_size=4, shuffle=False, num_workers=4)
test_dataloader = DataLoader(
    test_dataset, batch_size=4, shuffle=True, num_workers=4)

if __name__ == '__main__':

    for train_batch in train_dataloader:
        # print(len(train_batch))
        for label in train_batch[1]:
            cv2.imshow("img1", (label[0].numpy() * 255).astype('uint8'))
            cv2.imshow("img2", (label[1].numpy() * 255).astype('uint8'))
            cv2.waitKey(2000)

    for test_batch in test_dataloader:
        print(test_batch)
