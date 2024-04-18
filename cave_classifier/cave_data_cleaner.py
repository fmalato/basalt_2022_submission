import os

import numpy as np
import cv2

from tqdm import tqdm


if __name__ == '__main__':
    base_path = '/data/federima/cave_classifier/caves/'
    os.makedirs(base_path + 'filtered', exist_ok=True)
    cave_files = os.listdir(base_path)
    cave_files.remove('filtered')
    outliers = []
    outliers2 = []
    for n, image_file in tqdm(enumerate(cave_files)):
        file_path = base_path + image_file
        image = cv2.imread(file_path)
        if np.mean(image[:, :, 0]) + np.mean(image[:, :, 1]) + np.mean(image[:, :, 2]) / 3 > 128:
            outliers.append(n)
        elif np.mean(image[200:, :, 1]) > 80:
            outliers2.append(n)
        else:
            #cv2.imwrite(base_path + 'filtered/{}.jpg'.format(n), image)
            pass
    print(outliers)
    print(outliers2)
