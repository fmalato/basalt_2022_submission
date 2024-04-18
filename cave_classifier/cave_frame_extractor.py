import numpy as np
import os
import cv2
from tqdm import tqdm


if __name__ == '__main__':
    data_path = '/data/'
    videos_folder = os.listdir(data_path)
    videos = [x for x in videos_folder if x.rsplit(sep='.')[1] == 'mp4']
    n_frames = []
    progress_bar = tqdm(enumerate(videos))
    for i, file_name in progress_bar:
        video_capture = cv2.VideoCapture(data_path + file_name)
        success, image = video_capture.read()
        """current_frames = 0
        while(success):
            success, _ = video_capture.read()
            current_frames += 1"""
        n_frames.append(int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        video_capture.release()
        progress_bar.set_description('Extracting number of frames for video {}'.format(i))

    progress_bar = tqdm(enumerate(zip(videos, n_frames)))
    for i, (file_name, n) in progress_bar:
        if n > 10:
            video_capture = cv2.VideoCapture(data_path + file_name)
            sampled_frame = np.random.randint(0, int(n / 2))
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, sampled_frame)
            success, image = video_capture.read()
            cv2.imwrite('/data/federima/cave_classifier/not_caves/{}.jpg'.format(i), image)
            video_capture.release()
            progress_bar.set_description('Extracting non-cave frame from video {}'.format(i))
