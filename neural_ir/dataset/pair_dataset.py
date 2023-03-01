from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset
from neural_ir.utils.dataset_utils import read_pairs, read_triplets
import json
from ir_measures import read_trec_run


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

    HINT: - make sure to implement and use the functions defined in utils/dataset_utils.py
          - check out the documentation of ir_measures at https://ir-measur.es/en/latest/
          - the read_trec_run method returns a generator that yields the following object:
            `yield ScoredDoc(query_id=query_id, doc_id=doc_id, score=float(score))`
            (i.e., you can use pair.query_id and pair.doc_id by iterating through the generator)

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
        self.collection = {}
        with open(collection_path) as f:
            for line in f:
                doc_id, text = line.strip().split("\t")
                self.collection[doc_id] = text

        self.queries = {}
        with open(queries_path) as f:
            for line in f:
                query_id, text = line.strip().split("\t")
                self.queries[query_id] = text

        self.pairs = []
        for pair in read_trec_run(query_doc_pair_path):
            self.pairs.append((pair.query_id, pair.doc_id))

        if qrels_path:
            with open(qrels_path) as f:
                self.qrels = json.load(f)
        else:
            self.qrels = None

        self.top_k = top_k

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
