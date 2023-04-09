from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset
from neural_ir.utils.dataset_utils import read_pairs, read_triplets
import json
from ir_measures import read_trec_run
from collections import Counter

class PairDataset(Dataset):
    """
    PairDataset stores pairs of query and document needed to be score in the re-ranking step.
    Attributes
    ----------
    collection: dict
        a dictionary maps document id to text
    queries: dict
        a dictionary maps query id to text 
    pairs: list
        a list of (query, document) pairs for re-ranking
    qrels: dict
        a dictionary storing the ground-truth query relevancy.
    top_k: int
        an integer storing the number of documents to rerank per query

    """

    def __init__(
        self,
        collection_path: str,
        queries_path: str,
        query_doc_pair_path: str,
        qrels_path: str = None,
        top_k: int = 100,
    ):
        """
        Constructing PairDataset
        Parameters
        ----------
        collection_path: str
            path to a tsv file where each line store document id and text separated by a tab character 
        queries_path: str
            path to a tsv file where each line store query id and text separated by a tab character 
        query_doc_pair_path: str
            path to a trec run file (containing query-doc pairs) to re-rank
        qrels_path: str (optional)
            path to a qrel json file expected be formated as {query_id: {doc_id: relevance, ...}, ...}
        """
        self.collection = dict(read_pairs(collection_path))
        self.queries = dict(read_pairs(queries_path))
        with open(qrels_path, 'r') as r:
            self.qrels = json.load(r)
        self.pairs = []
        query_count = {}
        for pair in read_trec_run(query_doc_pair_path):
            q, d = pair.query_id, pair.doc_id
            if q not in query_count:
                query_count[q] = 1
            elif query_count[q] < top_k:
                query_count[q] += 1
            self.pairs.append((q, d))
        self.top_k = min([max(Counter(pair[0] for pair in self.pairs).values()), top_k])
    
    def __len__(self):
        """
        Return the number of pairs to re-rank
        """
        return len(self.pairs)

    def __getitem__(self, idx):
        """
        Return the idx-th pair of the dataset in the format of (qid, docid, query_text, doc_text)
        """
        query_id, doc_id = self.pairs[idx]
        query_text = self.queries[query_id]
        doc_text = self.collection[doc_id]
        return query_id, doc_id, query_text, doc_text
