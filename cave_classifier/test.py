import cv2
import torch
import argparse
import numpy as np
from network import CaveClassifier
from dataset import CaveDataset

from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', default=1, type=int)
    parser.add_argument('-f', '--save_frequency', default=5, type=int)
    parser.add_argument('-d', '--datadir')
    parser.add_argument('-b', '--batch_size', default=1, type=int)
    args = parser.parse_args()

    model = CaveClassifier(in_channels=3, input_width=320, input_height=320)
    model.load_state_dict(torch.load('/home/federima/basalt-2022-behavioural-cloning-baseline/cave_classifier/trained_models/010323_172104_10.pth'))
    model.cuda()
    data = CaveDataset(data_dir=args.datadir, validate=True, shuffle=True)
    train_loader = DataLoader(data, batch_size=args.batch_size, num_workers=8, pin_memory=True)
    data.set_valid_split(True)

    batch_size = 1
    progress_bar = tqdm(train_loader)
    correct = 0
    indices = []
    for i, batch in enumerate(progress_bar):
        with torch.no_grad():
            pred = model(batch['image'].cuda())
        pred.cpu().numpy()
        for offset, (y, yhat) in enumerate(zip(batch['label'], pred.cpu().numpy())):
            if y == np.argmax(yhat):
                correct += 1
            else:
                indices.append(i * args.batch_size + offset)
        progress_bar.set_description("Accuracy: {:.3f}%".format((correct / ((i+1) * batch_size)) * 100))
    #print(indices, len(indices))

    indices = list(np.random.randint(0, len(data), size=9))
    grid_size = 3
    np.random.shuffle(indices)
    fig, axs = plt.subplots(nrows=grid_size, ncols=grid_size, figsize=((3 * grid_size) + 1, (3 * grid_size) + 1))
    plt.tight_layout()
    plt.axis('off')
    for i, idx in enumerate(indices[:pow(grid_size, 2)]):
        batch = data[idx]
        with torch.no_grad():
            pred = model(batch['image'].cuda().unsqueeze(0))
        img = (batch['image'].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axs[int(i / grid_size), int(i % grid_size)].imshow(img)
        axs[int(i / grid_size), int(i % grid_size)].set_title('Label: {}, Pred: {}'.format(batch['label'], np.argmax(pred.cpu().numpy())))
        axs[int(i / grid_size), int(i % grid_size)].get_xaxis().set_visible(False)
        axs[int(i / grid_size), int(i % grid_size)].get_yaxis().set_visible(False)
    plt.show()
    fig.savefig('/home/federima/basalt-2022-behavioural-cloning-baseline/cave_classifier/figures/{}.png'.format(datetime.now().strftime("%d%m%y_%H%M%S")))
