import numpy as np
import torch
import torch.utils.data


class EpisodeSampler:
    def __init__(self, label, n_batch, n_cls, n_per, fix_seed=True):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per
        self.fix_seed = fix_seed

        label = np.array(label)
        self.m_ind = []
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

        if self.fix_seed:
            np.random.seed(0)
            self.cached_batches = []
            for i in range(self.n_batch):
                batch = []
                classes = np.random.choice(range(len(self.m_ind)), self.n_cls, False)
                for c in classes:
                    l = self.m_ind[c]
                    pos = np.random.choice(range(len(l)), self.n_per, False)
                    batch.append(l[pos])
                batch = torch.stack(batch).reshape(-1)
                self.cached_batches.append(batch)
            self.cached_batches = torch.stack(self.cached_batches)
            np.random.seed(0)

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            if self.fix_seed:
                batch = self.cached_batches[i_batch]
            else:
                batch = []
                classes = np.random.choice(range(len(self.m_ind)), self.n_cls, False)
                for c in classes:
                    l = self.m_ind[c]
                    pos = np.random.choice(range(len(l)), self.n_per, False)
                    batch.append(l[pos])
                batch = torch.stack(batch).reshape(-1)
            yield batch


class RepeatSampler:
    def __init__(self, dataset, batch_size, repeat):
        self.batch_size = batch_size//repeat
        self.repeat = repeat
        self.sampler = torch.utils.data.RandomSampler(dataset)
        self.drop_last = True

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                batch = batch * self.repeat
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size


class MultiTrans:
    def __init__(self, trans):
        self.trans = trans

    def __call__(self, x):
        out = []
        for trans in self.trans:
            out.append(trans(x))
        return out
