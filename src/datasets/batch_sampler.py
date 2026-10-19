import random

from torch.utils.data import Sampler


class LengthBucketBatchSampler(Sampler):
    def __init__(
        self, lengths, batch_size, bucket_multiplier=20, seed=42, drop_last=False
    ):
        self.lengths = list(lengths)
        self.batch_size = batch_size
        self.bucket_size = batch_size * bucket_multiplier
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        if self.drop_last:
            return len(self.lengths) // self.batch_size
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        indices = list(range(len(self.lengths)))
        rng.shuffle(indices)

        batches = []
        for start in range(0, len(indices), self.bucket_size):
            bucket = indices[start : start + self.bucket_size]
            bucket.sort(key=lambda index: self.lengths[index])
            for offset in range(0, len(bucket), self.batch_size):
                batch = bucket[offset : offset + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)

        rng.shuffle(batches)
        return iter(batches)
