import pytest
import torch

from neural_ir.models import SparseBiEncoder
from transformers import AutoTokenizer

@pytest.fixture
def sparse_encoder():
    model_name_or_dir="huawei-noah/TinyBERT_General_4L_312D"
    sparse = SparseBiEncoder(model_name_or_dir)
    return sparse

@pytest.fixture
def tokenizer():
    model_name_or_dir="huawei-noah/TinyBERT_General_4L_312D"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_dir)
    return tokenizer

def test_encode_text(sparse_encoder, tokenizer):
    texts =  ["first query", "first doc"]
    texts = tokenizer(texts, truncation=True, padding=True, return_tensors="pt")
    encoded_text = sparse_encoder.encode(**texts)
    assert type(encoded_text) == torch.Tensor
    assert encoded_text.shape == torch.Size([2, 30522])

def test_score_pairs(sparse_encoder, tokenizer):
    queries = ["first query", "second query"]
    docs = ["second query", "second doc"]
    queries = tokenizer(queries, truncation=True, padding=True, return_tensors="pt")
    docs = tokenizer(docs, truncation=True, padding=True, return_tensors="pt")
    scores, query_regularization, document_regularization = sparse_encoder.score_pairs(queries, docs, return_regularizer=True)
    assert type(scores) == torch.Tensor
    assert scores.shape == torch.Size([2])
    assert type(query_regularization) == torch.Tensor
    assert type(document_regularization) == torch.Tensor
    scores = sparse_encoder.score_pairs(queries, docs, return_regularizer=False)
    assert type(scores) == torch.Tensor
    assert scores.shape == torch.Size([2])


def test_forward(sparse_encoder, tokenizer):
    queries = ["queries 1", "queries 2"]
    pos_docs = ["pos for query 1", "pos for query 2"]
    neg_docs = ["neg for query 1", "neg for query 2"]
    queries = tokenizer(queries, truncation=True, padding=True, return_tensors="pt")
    pos_docs = tokenizer(pos_docs, truncation=True, padding=True, return_tensors="pt")
    neg_docs = tokenizer(neg_docs, truncation=True, padding=True, return_tensors="pt")
    loss, pos_scores, neg_scores = sparse_encoder(queries, pos_docs, neg_docs)
    assert type(loss) == torch.Tensor
    assert type(pos_scores) == torch.Tensor
    assert type(neg_scores) == torch.Tensor
    assert loss.shape == torch.Size([])
    assert pos_scores.shape == torch.Size([2])
    assert neg_scores.shape == torch.Size([2])