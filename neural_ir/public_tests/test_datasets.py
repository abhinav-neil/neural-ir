from neural_ir.dataset import PairDataset, TripletDataset
from collections import Counter
def test_pair_dataset():
    test_dataset = PairDataset(
        collection_path="data/tiny-data-for-test/collection.tsv",
        queries_path="data/tiny-data-for-test/test_queries.tsv",
        query_doc_pair_path="data/tiny-data-for-test/test_bm25.trec",
        qrels_path="data/tiny-data-for-test/test_qrels.json",
    )
    assert len(test_dataset) == 4
    assert type(test_dataset.collection) == dict
    assert type(test_dataset.queries) == dict
    assert type(test_dataset.pairs) == list
    assert len(test_dataset[0]) == 4
    assert max(Counter(pair[0] for pair in test_dataset.pairs).values()) <= test_dataset.top_k

def test_triplet_dataset():
    train_dataset = TripletDataset(
        collection_path="data/tiny-data-for-test/collection.tsv",
        queries_path="data/tiny-data-for-test/train_queries.tsv",
        train_triplets_path="data/tiny-data-for-test/train_triplets.tsv",
    )
    assert len(train_dataset) == 2274
    assert type(train_dataset.collection) == dict
    assert type(train_dataset.queries) == dict
    assert type(train_dataset.triplets) == list
    assert len(train_dataset[0]) == 3