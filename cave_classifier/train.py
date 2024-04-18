import torch
import argparse
from network import CaveClassifier
from dataset import CaveDataset

from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', default=1, type=int)
    parser.add_argument('-f', '--save_frequency', default=5, type=int)
    parser.add_argument('-d', '--datadir')
    parser.add_argument('-b', '--batch_size', default=1, type=int)
    args = parser.parse_args()

    model = CaveClassifier(in_channels=3, input_width=320, input_height=320)
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    # Maybe KL divergence?
    criterion = torch.nn.CrossEntropyLoss()
    valid_loss = torch.nn.CrossEntropyLoss()
    data = CaveDataset(data_dir=args.datadir, validate=True, shuffle=True)
    train_loader = DataLoader(data, batch_size=args.batch_size, num_workers=8, pin_memory=True)

    epoch_loss = []
    final_loss_plot = []
    final_loss_plot_valid = []
    for e in range(args.epochs):
        progress_bar = tqdm(train_loader)
        losses = []
        cum_loss = 0.0
        idx = 0
        avg_loss = 0.0
        for i, batch in enumerate(progress_bar):
            optimizer.zero_grad()
            pred = model(batch['image'].cuda())
            loss = criterion(pred, batch['label'].cuda())

            loss.backward()
            optimizer.step()
            cum_loss += loss.item()
            idx += 1
            avg_loss = cum_loss / idx
            losses.append(avg_loss)
            final_loss_plot.append(avg_loss)
            progress_bar.set_description(
                "Epoch {} - Loss: {:.5f} - Avg. Loss: {:.5f}".format(e + 1, loss.item(), avg_loss))
        epoch_loss.append(avg_loss)
    if (e + 1) % args.save_frequency == 0:
        save_path = '/home/federima/basalt-2022-behavioural-cloning-baseline/cave_classifier/trained_models/'
        model_name = datetime.now()
        torch.save(model.state_dict(),
                   save_path + '{}_{}.pth'.format(model_name.strftime("%d%m%y_%H%M%S"), e + 1))
