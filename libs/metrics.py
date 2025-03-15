import warnings
import numpy as np
import torch
from src.src.transformers import PreTrainedTokenizer
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from sklearn.metrics import f1_score
from rouge_score import rouge_scorer
from paddlenlp.metrics import Distinct


class Metric:
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.refs = []
        self.hyps = []
        self.tokenizer = tokenizer
        self.rouge_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        self.distinct = Distinct()

    def forward(self, refs, hyp):
        """Tokenize and store references and hypotheses."""
        self.refs.append([self.tokenizer.tokenize(ref) for ref in refs])
        self.hyps.append(self.tokenizer.tokenize(hyp))

    def get_bleu_k(self, k=4):
        """Calculate BLEU-k score using NLTK's corpus_bleu."""
        weights = tuple([1.0 / k] * k + [0.0] * (4 - k))
        try:
            return corpus_bleu(
                self.refs,
                self.hyps,
                weights=weights,
                smoothing_function=SmoothingFunction().method3,
            )
        except ZeroDivisionError:
            warnings.warn("Invalid BLEU score due to zero division.")
            return 0.0

    def get_distinct_k(self, k=2):
        """Tính chỉ số Distinct-k sử dụng paddlenlp."""
        predictions = [" ".join(hyp) for hyp in self.hyps]
        self.distinct.reset()
        for pred in predictions:
            self.distinct.add_inst(pred)
        return self.distinct.score()[f"distinct_{k}"]

    def get_unigram_f1(self):
        """Calculate Unigram F1 score using sklearn's f1_score."""
        hyp_tokens = [word for hyp in self.hyps for word in hyp]
        ref_tokens = [word for refs in self.refs for ref in refs for word in ref]
        if not hyp_tokens or not ref_tokens:
            return 0.0  # Tránh lỗi chia cho 0
        return f1_score(ref_tokens, hyp_tokens, average="macro")

    def get_rouge_l(self):
        """Calculate ROUGE-L using rouge_score library."""
        scores = [
            self.rouge_scorer.score(" ".join(ref[0]), " ".join(hyp))["rougeL"].fmeasure
            for ref, hyp in zip(self.refs, self.hyps)
        ]
        return np.mean(scores)


def get_ppl_value(lm_logits: torch.Tensor, labels: torch.Tensor) -> float:
    loss = torch.functional.F.cross_entropy(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1), reduction='none')
    loss = loss.view(labels.size(0), labels.size(1))
    label_size = torch.sum(labels.ne(-100), dim=1).type_as(loss)
    ppl_value = torch.exp(torch.mean(torch.sum(loss, dim=1).float() / label_size.float()))

    return ppl_value, label_size
    
