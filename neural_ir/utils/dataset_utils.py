from pathlib import Path
from tqdm import tqdm

def read_pairs(path: str):
    """
    Read tab-delimited pairs from file.
    Parameters
    ----------
    path: str 
        path to the input file
    Returns
    -------
        a list of pair tuple
    """
    pairs = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in tqdm(f.readlines(), desc=f'reading pairs from {Path(path).name}'):
            qid, did = line.strip().split('\t')
            pairs.append((qid, did))
    return pairs

def read_triplets(path: str):
    """
    Read tab-delimited triplets from file.
    Parameters
    ----------
    path: str 
        path to the input file
    Returns
    -------
        a list of triplet tuple
    """
    triplets = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in tqdm(f.readlines(), desc=f'reading triplets from {Path(path).name}'):
            qid, pos_id, neg_id = line.strip().split('\t')
            triplets.append((qid, pos_id, neg_id))
    return triplets
