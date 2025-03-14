import os
import pickle
import numpy as np
import torch
from collections import Counter
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer


def create_folder(folder_name: str):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
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


def weight_text_by_hybrid(words: str, arousal_scores: torch.Tensor):
    words = norm(words).split()
    word_counts = Counter(words)
    tf_weights = torch.tensor([word_counts[word] / len(words) for word in words])
    word_lengths = torch.tensor([len(word) for word in words])

    tf_weights /= tf_weights.sum()
    word_lengths /= word_lengths.sum()
    arousal_scores /= arousal_scores.sum() + 1e-6 

    weights = (tf_weights + word_lengths + arousal_scores) / 3

    return torch.tensor(weights, dtype=torch.float32) 


def weight_text_by_tf_ids(words: str) -> torch.Tensor:
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([words])
    tfidf_scores = np.array(tfidf_matrix.todense()).flatten()
    total_tfidf = np.sum(tfidf_scores)

    if total_tfidf == 0:
        return torch.zeros_like(torch.tensor(tfidf_scores)) 
    
    weights = tfidf_scores / total_tfidf

    return torch.tensor(weights, dtype=torch.float32)


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

