# coding=utf-8
import math
import random
from torch.utils.data import Sampler


class BucketSampler(Sampler):
    """
    this sampler will sort data by sequence length
    """
    def __init__(self, lens, bucket_size, batch_size,
                 droplast=False, shuffle=True):
        self._lens = lens
        self._batch_size = batch_size
        self._bucket_size = bucket_size
        self._droplast = droplast
        self._shuf = shuffle

    def __iter__(self):
        ids = list(range(len(self._lens)))
        if self._shuf:
            random.shuffle(ids)
        buckets = [sorted(ids[i:i+self._bucket_size], key=lambda i: self._lens[i], reverse=True) for i in range(0, len(ids), self._bucket_size)]
        batches = [bucket[i:i+self._batch_size] for bucket in buckets for i in range(0, len(bucket), self._batch_size)]
        if self._droplast:
            batches = [batch for batch in batches if len(batch) == self._batch_size]
        if self._shuf:
            random.shuffle(batches)
        return iter(batches)

    def __len__(self):
        bucket_sizes = ([self._bucket_size] * (len(self._lens) // self._bucket_size) + [len(self._lens) % self._bucket_size])
        if self._droplast:
            return sum(s//self._batch_size for s in bucket_sizes)
        else:
            return sum(math.ceil(s/self._batch_size) for s in bucket_sizes)