"""Script to generate and stores data into local data directory"""

import argparse
import math
from pathlib import Path
from src.utils import IterableSequenceAlternation
import numpy as np


def parse_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_unique', type=int, default=20)
    parser.add_argument('--cycle_len', type=int, default=4)
    parser.add_argument('--data_dir', default='data')
    parser.add_argument('--train_dir', default='train')
    parser.add_argument('--test_dir', default='test')
    # parser.add_argument('--train_filename', default='train.csv')
    # parser.add_argument('--test_filename', default='test.csv')
    parser.add_argument('--full_filename', default='full.csv')
    parser.add_argument('--train_prop', default=.9)
    args = parser.parse_args()
    return args


def setup_file_paths(args):

    global PARENT_DIR, DATA_DIR, TRAIN_DIR, TEST_DIR, TRAIN_FILE, TEST_FILE, \
        FULL_FILE
    PARENT_DIR = Path(__file__).parents[1]
    DATA_DIR = PARENT_DIR / args.data_dir
    TRAIN_DIR = DATA_DIR / args.train_dir
    TEST_DIR = DATA_DIR / args.test_dir
    # TRAIN_FILE = TRAIN_DIR / args.train_filename
    # TEST_FILE = TEST_DIR / args.test_filename
    FULL_FILE = PARENT_DIR / args.data_dir / args.full_filename


if __name__ == '__main__':
    args = parse_args()
    setup_file_paths(args)

    dset = IterableSequenceAlternation(args.num_unique, args.cycle_len)
    print(f'Creating new dataset with settings: {vars(args)}')
    with open(FULL_FILE, 'w') as f:
        # write header
        seq_len = args.cycle_len * 4
        x_cols_str = ','.join([f'x{i}' for i in range(1, seq_len + 1)])
        target_cols_str = ','.join([f'y{i}' for i in range(1, seq_len + 1)])
        f.write(x_cols_str + ',' + target_cols_str + '\n')

        # write data
        for i, (x, target) in enumerate(iter(dset), start=1):
            x = map(str, x)
            target = map(str, target)
            f.write(','.join(x)+',')
            f.write(','.join(target)+'\n')

        print(f'Complete dataset written to {FULL_FILE}')
        print(f'Number of sentences output: {i}')
        assert i == math.perm(args.num_unique, args.cycle_len+1), \
            'Incorrect number of permutations'

        num_obs = i

        # select and output train/test indices
        print('Preparing training data...')
        num_train = int(num_obs * args.train_prop)
        train_idx = np.random.choice(range(num_obs), size=num_train,
                                     replace=False)
        test_idx = np.array(list(set(range(num_obs)) - set(train_idx)))

        np.save(TRAIN_DIR/'train_indices', train_idx)
        np.save(TEST_DIR/'test_indices', test_idx)
