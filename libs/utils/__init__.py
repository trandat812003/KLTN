import os
import pickle
import numpy as np
import torch
from collections import Counter
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer


def create_folder(folder_name: str):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name, exist_ok=True)
        return

def read_file(file_name: str) -> list[str]:
    try:
        with open(file_name, 'r', encoding='utf-8') as f:
            return f.readlines()
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_name}")
    

def load_file_pickle(root_file: str, file_name: str):
    if not os.path.exists(root_file):
        create_folder(folder_name=root_file)

    file_name = os.path.join(root_file, file_name)
    if not os.path.exists(file_name):
        return None, None
    
    with open(file_name, 'rb') as f:
        reader = pickle.load(f)
    
    return reader['data_list'], reader['inputs']


def save_file_pickle(file_name: str, data: any):
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)


def cut_seq_to_eos(sentence, eos, remove_id=None):
    if remove_id is None:
        remove_id = [-1]
    sent = []
    for s in sentence:
        if s in remove_id:
            continue
        if s != eos:
            sent.append(s)
        else:
            break
    return sent


def norm(s: str) -> str:
    return ' '.join(s.strip().split())


def padding(inputs: torch.Tensor, max_len: int, pad: float) -> torch.Tensor:
    if inputs.shape[0] >= max_len:
        return inputs[:max_len]
    
    pad_size = max_len - inputs.shape[0]
    return F.pad(inputs, (0, pad_size), value=pad)

