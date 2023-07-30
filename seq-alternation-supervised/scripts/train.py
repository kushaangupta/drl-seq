"""Script to train networks.
Assumes you've generated the train/test indices using `generate_data.py`
"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.utils.tensorboard as tb

from src.utils import SequenceAlternation
from src.models import RNNNet


def parse_args() -> argparse.Namespace: 
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_unique', type=int, default=20)
    parser.add_argument('--cycle_len', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=3500)
    parser.add_argument('--lr', type=float, default=.0005)
    parser.add_argument('--hidden_size', type=int, default=30)
    parser.add_argument('--log_dir', default='logs')
    parser.add_argument('--data_dir', default='data')
    parser.add_argument('--train_dir', default='train')
    parser.add_argument('--test_dir', default='test')
    parser.add_argument('--full_filename', default='full.csv')
    parser.add_argument('--logging', action='store_true', default=False)
    args = parser.parse_args()
    return args


def setup_file_paths(args):

    global PARENT_DIR, DATA_DIR, TRAIN_DIR, TEST_DIR, TRAIN_FILE, TEST_FILE, \
        ALL_FILE, LOGGING_DIR, FULL_FILE

    PARENT_DIR = Path(__file__).parents[1]
    DATA_DIR = PARENT_DIR / args.data_dir
    TRAIN_DIR = DATA_DIR / args.train_dir
    TEST_DIR = DATA_DIR / args.test_dir
    # TRAIN_FILE = TRAIN_DIR / args.train_filename
    # TEST_FILE = TEST_DIR / args.test_filename
    ALL_FILE = PARENT_DIR / args.data_dir / args.full_filename
    LOGGING_DIR = PARENT_DIR / Path(args.log_dir)
    FULL_FILE = PARENT_DIR / args.data_dir / args.full_filename

# def tb_logging(train_model):
#         def log_train(*args, **kwargs):
#             train_model(*args, **kwargs)

#     return log_train


if __name__ == '__main__':
    args = parse_args()
    setup_file_paths(args)

    # logging setup
    if args.logging:
        print('Logging to tensorboard!')
        now = datetime.now().strftime('%m-%d %H:%M')
        writer = tb.SummaryWriter(log_dir=LOGGING_DIR/now, flush_secs=1)

    # data setup
    train_idx = np.load(TRAIN_DIR/'train_indices.npy')
    test_idx = np.load(TEST_DIR/'test_indices.npy')
    dset = SequenceAlternation(str(FULL_FILE), args.num_unique, args.cycle_len)
    train_loader = DataLoader(dset, batch_size=args.batch_size,
                              sampler=train_idx)
    dev_loader = DataLoader(dset, batch_size=args.batch_size,
                            sampler=test_idx)

    # model setup
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available()
              else "cpu")
    print(f'Using {device}!')

    model = RNNNet(seq_len=1, hidden_size=args.hidden_size,
                   num_layers=1, batch_size=args.batch_size)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # training loop
    train_step = 0
    epoch = 0
    dev_improv = np.inf
    prev_dev_loss = np.inf
    while dev_improv >= 0:
        model.train()
        for x_batch, y_batch in train_loader:
            # print('y_batch:', y_batch.shape)
            # print('x_batch', x_batch.shape)
            optimizer.zero_grad()
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            target = model.forward(x_batch)
            # print(outputs.shape, '\n')
            train_loss = criterion(target, y_batch)
            train_loss.backward()
            optimizer.step()

            if train_step % 20 == 0:
                print(f'train loss: {train_loss.item()}')
                if args.logging:
                    writer.add_scalar('train/loss', train_loss.item(),
                                      global_step=train_step)
            train_step += 1

        eval_losses = []
        model.eval()
        for x_batch, y_batch in dev_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            target = model.forward(x_batch) 
            eval_loss = criterion(target, y_batch)
            eval_losses.append(eval_loss.item())
            print(f'eval loss {eval_loss.item()}')

        current_dev_loss = sum(eval_losses)/len(eval_losses)
        dev_improv = prev_dev_loss - current_dev_loss
        prev_dev_loss = current_dev_loss
        if args.logging:
            writer.add_scalar('dev/loss', current_dev_loss, global_step=epoch)
        epoch += 1
