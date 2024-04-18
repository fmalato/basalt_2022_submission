import os

import numpy as np
import cv2

from tqdm import tqdm


if __name__ == '__main__':
    base_path = '/data/federima/cave_classifier/'
    caves_path = 'caves/filtered/'
    non_caves_path = 'not_caves/'
    sample_width = 320
    sample_height = 320
    train_val_split = 0.9
    image_width = 360
    image_height = 640
    # -1 accounts for 0-indexing
    starting_index_x = (int(image_width / 2) - int(sample_width / 2)) - 1
    ending_index_x = (int(image_width / 2) + int(sample_width / 2)) - 1
    starting_index_y = (int(image_height / 2) - int(sample_height / 2)) - 1
    ending_index_y = (int(image_height / 2) + int(sample_height / 2)) - 1
    # Create directories
    os.makedirs('/data/federima/cave_classifier/train/caves/', exist_ok=True)
    os.makedirs('/data/federima/cave_classifier/train/not_caves/', exist_ok=True)
    os.makedirs('/data/federima/cave_classifier/valid/caves/', exist_ok=True)
    os.makedirs('/data/federima/cave_classifier/valid/not_caves/', exist_ok=True)
    # Train caves
    cave_files = os.listdir(base_path + caves_path)
    for f in tqdm(cave_files[: int(train_val_split * len(cave_files))]):
        image = cv2.imread(base_path + caves_path + f)
        image = image[starting_index_x:ending_index_x, starting_index_y:ending_index_y, :]
        assert image.shape[0] == sample_width and image.shape[1] == sample_height, 'mismatching image dimension: {}'.format(image.shape)
        cv2.imwrite('/data/federima/cave_classifier/train/caves/{}'.format(f), image)
    # Val caves
    for f in tqdm(cave_files[int(train_val_split * len(cave_files)):]):
        image = cv2.imread(base_path + caves_path + f)
        image = image[starting_index_x:ending_index_x, starting_index_y:ending_index_y, :]
        assert image.shape[0] == sample_width and image.shape[
            1] == sample_height, 'mismatching image dimension: {}'.format(image.shape)
        cv2.imwrite('/data/federima/cave_classifier/valid/caves/{}'.format(f), image)
    # Train not caves
    not_cave_files = os.listdir(base_path + non_caves_path)
    for f in tqdm(not_cave_files[: int(train_val_split * len(cave_files))]):
        image = cv2.imread(base_path + non_caves_path + f)
        image = image[starting_index_x:ending_index_x, starting_index_y:ending_index_y, :]
        assert image.shape[0] == sample_width and image.shape[
            1] == sample_height, 'mismatching image dimension: {}'.format(image.shape)
        cv2.imwrite('/data/federima/cave_classifier/train/not_caves/{}'.format(f), image)
    # Val not caves
    for f in tqdm(not_cave_files[int(train_val_split * len(cave_files)):]):
        image = cv2.imread(base_path + non_caves_path + f)
        image = image[starting_index_x:ending_index_x, starting_index_y:ending_index_y, :]
        assert image.shape[0] == sample_width and image.shape[
            1] == sample_height, 'mismatching image dimension: {}'.format(image.shape)
        cv2.imwrite('/data/federima/cave_classifier/valid/not_caves/{}'.format(f), image)
