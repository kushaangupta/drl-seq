from itertools import chain, permutations, repeat
import linecache
import torch
from torch.utils.data import IterableDataset, Dataset


class IterableSequenceAlternation(IterableDataset):
    """Memory efficient sequence dataset for generating the full data"""
    def __init__(self, num_unique: int, cycle_len: int):
        self.num_unique, self.cycle_len = num_unique, cycle_len
        self.perms = permutations(range(1, num_unique+1), cycle_len+1)

    def __next__(self):
        perm = list(next(self.perms))
        first_cycle = perm[:self.cycle_len]
        second_cycle = perm[:self.cycle_len-1] + [perm[-1]]

        input_seq = chain(first_cycle,
                          second_cycle,
                          repeat(0, 2*self.cycle_len))

        output_seq = chain(first_cycle,
                           second_cycle,
                           first_cycle,
                           second_cycle)

        return list(input_seq), list(output_seq)
        # return torch.Tensor(list(input_seq)), torch.Tensor(list(output_seq))

    def __iter__(self):
        return self


class SequenceAlternation(Dataset):
    """Sequence dataset used for training"""
    def __init__(self, dset_path, num_unique: int, cycle_len: int, emb=False):
        self.dset_path = dset_path
        self.num_unique = num_unique
        self.cycle_len = cycle_len
        self.emb = emb

    def __getitem__(self, idx):
        line = linecache.getline(self.dset_path, idx+2) \
                .strip()    \
                .split(',')
        seq_len = len(line) // 2
        input_seq = list(map(int, line[:seq_len]))
        input_seq = torch.tensor(input_seq, dtype=torch.float32)
        output_seq = list(map(int, line[seq_len:]))
        output_seq = torch.tensor(output_seq, dtype=torch.float32)

        # (batched) input shape should be: (N, L, input_dim)
        if self.emb is False:
            input_seq = torch.unsqueeze(input_seq, -1)
            output_seq = torch.unsqueeze(output_seq, -1)

        return input_seq, output_seq

    def validate_saved_dataset(self):
        """TODO"""
        pass
        
    def __len__(self):
        # no choice but to read entire file
        with open(self.dset_path, 'rbU') as f:
            num_lines = sum(1 for _ in f)
            num_obs = num_lines - 1
        return num_obs


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    dset = IterableSequenceAlternation(num_unique=10, cycle_len=4)
    loader = DataLoader(dset)
    
    # for i, (x, y) in enumerate(loader):
    #     print(i)
    #     print(x)
    #     print(y)
    #     break