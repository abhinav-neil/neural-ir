import pytest
import torch
from neural_ir.models.sparse_encoder import L1Regularizer
def test_regularizer():
    l1_regularizer = L1Regularizer()
    input_reps = torch.rand(10,2)
    output = l1_regularizer(input_reps)
    assert torch.allclose(output, torch.zeros(10))
    assert l1_regularizer.current_step == 1
    assert l1_regularizer.current_alpha > 0
    
