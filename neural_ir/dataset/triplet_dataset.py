from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset

from neural_ir.utils.dataset_utils import read_pairs, read_triplets


class TripletDataset(Dataset):
    """
    TripletDataset for contrastive training. This dataset consists of a list of (query, positive document, negative document) triplets.
    During training, the model would be optimize to produce high scores for (query, positive) pairs and low scores for (query, negative) pairs.
    Attributes
    ----------
    collection: dict 
        a dictionary that maps document id to document text
    queries: dict
        a dictionary that maps query id to query text
    triplets: list
        a list of id triplets (query_id, pos_id, neg_id)

    HINT: - make sure to implement and use the functions defined in utils/dataset_utils.py
    """

    def __init__(
        self, collection_path: str, queries_path: str, train_triplets_path: str
    ):
        """
        Constructing a TripletDataset
        Parameters
        ----------
        collection_path: str
            path to a tsv file,  each line has a document id and document text separated by a tab character
        queries_path: str
            path to a tsv file,  each line has a query id and query text separated by a tab character
        train_triplets_path: str
            path to a tsv file,  each line has a query id, positive id and negative id separated by a tab character
        """
        self.collection = {}
        with open(collection_path, "r") as f:
            for line in f:
                id_, text = line.rstrip("\n").split("\t")
                self.collection[id_] = text

        self.queries = {}
        with open(queries_path, "r") as f:
            for line in f:
                id_, text = line.rstrip("\n").split("\t")
                self.queries[id_] = text

        self.triplets = read_triplets(train_triplets_path)

    def __len__(self):
        """
        Return the number of triplets
        """
        return len(self.triplets)

    def __getitem__(self, idx):
        """
        Get text contents of the idx-th triplet. 
        Parameters
        ----------
        idx: int 
            the index of the triplet to return
        Returns:
        tuple:
            (query_text, pos_text, neg_text) 
        """
        qid, pid, nid = self.triplets[idx]
        q_text = self.queries[qid]
        p_text = self.collection[pid]
        n_text = self.collection[nid]
        return q_text, p_text, n_text
