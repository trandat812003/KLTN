# coding=utf-8
import pickle
from functools import partial
from torch.utils.data import DataLoader


from .bucket_sampler import BucketSampler


class BucketingDataLoader(object):
    def __init__(self, toker, feature_dataset, batch_size,
                 bucket=100, shuffle=True, **kwargs):
        assert 'inputter_name' in kwargs
        assert 'config_name' in kwargs
        assert 'data_name' in kwargs
        assert 'knowledge_name' in kwargs
        inputter_name = kwargs.pop('inputter_name')
        config_name = kwargs.pop('config_name')
        data_name = kwargs.pop('data_name')
        knowledge_name = kwargs.pop('knowledge_name')
        with open(f'./DATA/{inputter_name}.{config_name}.{data_name}.{knowledge_name}/data.pkl', 'rb') as f:
            self.data = pickle.load(f)
        self.toker = toker
        self.data_name = data_name
        self.knowledge_name = knowledge_name
        self.feature_dataset = feature_dataset
        self.batch_size = batch_size
        self.bucket_size = bucket * batch_size
        self.shuffle = shuffle

    def __iter__(self):
        trunc_chunk = []
        lens = []
        for feat in self.data:
            trunc_chunk.append(feat)
            lens.append(feat.input_len)

        dataset = self.feature_dataset(trunc_chunk)
        sampler = BucketSampler(
            lens, 
            self.bucket_size, 
            self.batch_size,
            droplast=True, 
            shuffle=self.shuffle
        )
        loader = DataLoader(
            dataset, 
            batch_sampler=sampler,
            num_workers=0,
            collate_fn=partial(
                self.feature_dataset.collate, 
                toker=self.toker, 
                data_name=self.data_name, 
                knowledge_name=self.knowledge_name
            )
        )
        yield from loader

    def __len__(self):
        return len(self.data)