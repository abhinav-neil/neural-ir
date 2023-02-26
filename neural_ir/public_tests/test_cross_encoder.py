import pytest
import torch

from neural_ir.models import CrossEncoder
from transformers import AutoTokenizer


@pytest.fixture
def cross_encoder():
    model_name_or_dir = "huawei-noah/TinyBERT_General_4L_312D"
    ce = CrossEncoder(model_name_or_dir)
    return ce


@pytest.fixture
def tokenizer():
    model_name_or_dir = "huawei-noah/TinyBERT_General_4L_312D"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_dir)
    return tokenizer


def test_score_pairs(cross_encoder, tokenizer):
    pairs = [
        ["first query", "first doc"],
        ["second query", "second doc"]
    ]
    pairs = tokenizer(pairs, truncation=True, padding=True, return_tensors="pt")
    scores = cross_encoder.score_pairs(pairs)
    assert type(scores) == torch.Tensor
    assert scores.shape == torch.Size([2])

def test_forward(cross_encoder, tokenizer):
    pos_pairs = [
        ["first query", "pos doc"],
        ["second query", "pos doc"]
    ]
    neg_pairs = [
        ["first query", "neg doc"],
        ["second query", "neg doc"]
    ]
    pos_pairs = tokenizer(pos_pairs, truncation=True, padding=True, return_tensors="pt")
    neg_pairs = tokenizer(neg_pairs, truncation=True, padding=True, return_tensors="pt")
    loss, pos_scores, neg_scores = cross_encoder(pos_pairs, neg_pairs)
    assert type(loss) == torch.Tensor
    assert type(pos_scores) == torch.Tensor
    assert type(neg_scores) == torch.Tensor
    assert loss.shape == torch.Size([])
    assert pos_scores.shape == torch.Size([2])
    assert neg_scores.shape == torch.Size([2])