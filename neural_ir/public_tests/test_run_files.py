import ir_measures
from ir_measures import *
from pathlib import Path
import json
import pytest


@pytest.fixture
def dev_qrels():
    return json.load(open("data/dev_qrels.json"))


def test_ce_run_file_available(dev_qrels):
    path = Path("output/ce/test_run.trec")
    assert (
        path.is_file()
    ), "cross encoder's test run file is not available at output/ce/test_run.trec"
    run_file = ir_measures.read_trec_run(str(path))
    results = ir_measures.calc_aggregate([MRR @ 10, R @ 1000], dev_qrels, run_file)
    metrics = {k: v for k, v in results.items()}
    mrr = metrics[MRR @ 10]
    assert mrr > 0


def test_dense_run_file_available(dev_qrels):
    path = Path("output/dense/test_run.trec")
    assert (
        path.is_file()
    ), "dense's test run file is not available at output/dense/test_run.trec"
    path = Path("output/ce/test_run.trec")
    assert (
        path.is_file()
    ), "cross encoder's test run file is not available at output/ce/test_run.trec"
    run_file = ir_measures.read_trec_run(str(path))
    results = ir_measures.calc_aggregate([MRR @ 10, R @ 1000], dev_qrels, run_file)
    metrics = {k: v for k, v in results.items()}
    mrr = metrics[MRR @ 10]
    assert mrr > 0


def test_sparse_run_file_available(dev_qrels):
    path = Path("output/sparse/test_run.trec")
    assert (
        path.is_file()
    ), "dense's test run file is not available at output/sparse/test_run.trec"
    run_file = ir_measures.read_trec_run(str(path))
    results = ir_measures.calc_aggregate([MRR @ 10, R @ 1000], dev_qrels, run_file)
    metrics = {k: v for k, v in results.items()}
    mrr = metrics[MRR @ 10]
    assert mrr > 0
