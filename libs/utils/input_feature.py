import torch
from torch.nn.utils.rnn import pad_sequence
from transformers.tokenization_utils import PreTrainedTokenizer
from typing import List
from libs.config import Config


class InputFeature(object):
    def __init__(self, input_ids, decoder_input_ids, labels):
        self.input_ids = input_ids
        self.input_length = len(input_ids)

        self.decoder_input_ids = decoder_input_ids
        self.decoder_input_length = len(decoder_input_ids)
        self.labels = labels

        self.input_len = self.input_length + self.decoder_input_length

    @staticmethod
    def collate(features: List['InputFeature'], tokenizer: PreTrainedTokenizer):
        pad = tokenizer.pad_token_id
        if pad is None:
            pad = tokenizer.eos_token_id
            assert pad is not None, 'either pad_token_id or eos_token_id should be provided'
        
        input_ids = pad_sequence(
            [torch.tensor(f.input_ids, dtype=torch.long) for f in features],
            batch_first=True, 
            padding_value=pad
        )
        attention_mask = pad_sequence(
            [torch.tensor([1.] * f.input_length, dtype=torch.float) for f in features],
            batch_first=True, 
            padding_value=0.
        )
        decoder_input_ids = pad_sequence(
            [torch.tensor(f.decoder_input_ids, dtype=torch.long) for f in features],
            batch_first=True, 
            padding_value=pad
        )
        labels = pad_sequence(
            [torch.tensor(f.labels, dtype=torch.long) for f in features],
            batch_first=True, 
            padding_value=-100
        )
        
        if Config.DATA_NAME == 'esconv':
            strat_id = torch.tensor([f.labels[0] for f in features], dtype=torch.long) - len(tokenizer) + 8
        elif Config.DATA_NAME == 'mi':
            strat_id = torch.tensor([f.labels[0] for f in features], dtype=torch.long) - len(tokenizer) + 10
        
        if Config.KNOWLEDGE_NAME == 'basic':
            strat_id += 5
        elif Config.KNOWLEDGE_NAME == 'bm25':
            strat_id += 1
        elif Config.KNOWLEDGE_NAME == 'oracle':
            strat_id += 6
        elif Config.KNOWLEDGE_NAME in ['sbert','graph']:
            strat_id += 8
        
        res = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'decoder_input_ids': decoder_input_ids,
            'labels': labels,
            'strat_id': strat_id,
        }
        
        return res