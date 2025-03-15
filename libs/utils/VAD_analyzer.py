import pandas as pd
import torch
from collections import Counter

from libs.utils import norm


class VADAnalyzer:
    _VAD = pd.read_csv("./DATA/NRC-VAD-Lexicon.csv", keep_default_na=False)
    _VAD.set_index("Word", inplace=True)

    @classmethod
    def _get_vad_scores(cls, text: str) -> tuple[torch.Tensor, list]:
        vad_scores = []
        words = []
        for word in cls._VAD.index:
            start = 0
            while start < len(text):
                idx = text.find(word, start)
                if idx == -1:
                    break
                valence = cls._VAD.loc[word, "Valence"]
                arousal = cls._VAD.loc[word, "Arousal"]
                dominance = cls._VAD.loc[word, "Dominance"]
                vad_scores.append(
                    (
                        idx,
                        torch.tensor(
                            [valence, arousal, dominance], dtype=torch.float32
                        ),
                    )
                )
                words.append((idx, word))
                start = idx + len(word)
        if not vad_scores:
            return torch.tensor([], dtype=torch.float32), words
        vad_scores.sort(key=lambda x: x[0])
        words.sort(key=lambda x: x[0])
        vad_scores = torch.stack([vad_score[1] for vad_score in vad_scores])

        return vad_scores, words

    @classmethod
    def compute_weighted_vad(cls, text: str) -> tuple[float, float, float]:
        vad_scores, words = cls._get_vad_scores(text)
        if vad_scores.shape[0] == 0:
            return (0.0, 0.0, 0.0)
        weights = weight_text_by_hybrid(words, vad_scores[:, 1])
        if vad_scores.shape[0] == 0 or weights.shape[0] == 0:
            return (0.0, 0.0, 0.0)

        weights = weights / weights.sum()
        weighted_vad = (vad_scores * weights.unsqueeze(1)).sum(dim=0)

        return tuple(weighted_vad.tolist())
    
    @classmethod
    def categorize_value(value):
        if value <= 0.33:
            return "low"
        elif value <= 0.66:
            return "medium"
        else:
            return "high"


def weight_text_by_hybrid(words: list[str], arousal_scores: torch.Tensor):
    if not words:
        return torch.tensor([], dtype=torch.float32)
    tf_weights = weight_text_by_tf_ids(words)
    word_lengths = torch.tensor([len(word) for word in words], dtype=torch.float32)
    word_lengths /= word_lengths.sum()
    arousal_scores /= arousal_scores.sum() + 1e-6

    weights = (tf_weights + word_lengths + arousal_scores) / 3

    return torch.tensor(weights, dtype=torch.float32)


def weight_text_by_tf_ids(words: list[str]) -> torch.Tensor:
    word_counts = Counter(words)
    tf_weights = torch.tensor(
        [word_counts[word] / len(words) for word in words], dtype=torch.float32
    )

    tf_weights /= tf_weights.sum()

    return torch.tensor(tf_weights, dtype=torch.float32)
