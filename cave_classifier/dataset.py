import os
import cv2
import numpy as np
import torch

from torch.utils.data import Dataset


class CaveDataset(Dataset):

    def __init__(self, data_dir, validate=True, shuffle=True):
        super().__init__()
        self.data_dir = data_dir
        self.shuffle = shuffle
        cave = [('train/caves/{}'.format(x), 1) for x in os.listdir(self.data_dir + 'train/caves')]
        not_cave = [('train/not_caves/{}'.format(x), 0) for x in os.listdir(self.data_dir + 'train/not_caves')]
        self.train_data = cave + not_cave
        if self.shuffle:
            np.random.shuffle(self.train_data)
        self.validate = validate
        if self.validate:
            cave = [('valid/caves/{}'.format(x), 1) for x in os.listdir(self.data_dir + 'valid/caves')]
            not_cave = [('valid/not_caves/{}'.format(x), 0) for x in os.listdir(self.data_dir + 'valid/not_caves')]
            self.valid_data = cave + not_cave
            if self.shuffle:
                np.random.shuffle(self.valid_data)
        else:
            self.valid_data = None
        self._valid_split = False

    def __len__(self):
        if self._valid_split and self.validate:
            return len(self.valid_data)
        else:
            return len(self.train_data)

    def __getitem__(self, item):
        if self._valid_split:
            image = cv2.imread(self.data_dir + self.valid_data[item][0]) / 255
            image = torch.FloatTensor(image)
            image = image.permute(2, 0, 1)
            label = self.valid_data[item][1]
        else:
            image = cv2.imread(self.data_dir + self.train_data[item][0]) / 255
            image = torch.FloatTensor(image)
            image = image.permute(2, 0, 1)
            label = self.train_data[item][1]

        return {'image': image, 'label': label}

    def set_valid_split(self, value):
        self._valid_split = value


if __name__ == '__main__':
    data = CaveDataset(data_dir='/data/federima/cave_classifier/', validate=True)
    idx = 5
    batch = data[idx]
    data.set_valid_split(True)
    img, label = data[idx]
