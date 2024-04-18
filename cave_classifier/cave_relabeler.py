import os
import cv2


if __name__ == '__main__':
    images = os.listdir('/data/federima/cave_classifier/valid/false_positives')
    count = 6000
    for img in images:
        image = cv2.imread('/data/federima/cave_classifier/valid/false_positives/' + img)
        cv2.imwrite('/data/federima/cave_classifier/valid/not_caves/{}.jpg'.format(count), image)
        count += 1
