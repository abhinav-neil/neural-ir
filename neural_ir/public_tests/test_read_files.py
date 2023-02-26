from neural_ir.utils.dataset_utils import read_pairs, read_triplets
def test_read_pairs():
    pairs = read_pairs("data/tiny-data-for-test/test_queries.tsv")
    assert len(pairs) == 10
    assert len(pairs[0]) == 2
    assert pairs[0][0] == "868487"

def test_read_triplets():
    triplets = read_triplets("data/tiny-data-for-test/train_triplets.tsv")
    assert len(triplets) == 2274
    assert len(triplets[0]) == 3
    assert triplets[0][0] == "399998"