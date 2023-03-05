import argparse
from neural_ir.index import Faiss
from neural_ir.models.your_creativity import MyDenseBiEncoder
from neural_ir.utils import write_trec_run
from neural_ir.utils.dataset_utils import read_pairs
from transformers import AutoTokenizer
from tqdm import tqdm
import torch
from collections import defaultdict

parser = argparse.ArgumentParser(description="Ranking with BiEncoder")
parser.add_argument(
    "--c", type=str, default="data/collection.tsv", help="path to document collection"
)
parser.add_argument(
    "--q", type=str, default="data/test_queries.tsv", help="path to queries"
)
parser.add_argument(
    "--device", type=str, default="cuda", help="device to run inference"
)
parser.add_argument("--bs", type=int, default=16, help="batch size")
parser.add_argument(
    "--checkpoint",
    default="output/your_creativity/model",
    type=str,
    help="path to model checkpoint",
)
parser.add_argument(
    "--o",
    type=str,
    default="output/your_creativity/test_run.trec",
    help="path to output run file",
)
args = parser.parse_args()

docs = read_pairs(args.c)
queries = read_pairs(args.q)

tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
model = MyDenseBiEncoder.from_pretrained(args.checkpoint).to(args.device)
model.eval()
query_embs = []
docs_embs = []
doc_ids = []
for idx in tqdm(
    range(0, len(docs), args.bs), desc="Encoding documents", position=0, leave=True
):
    batch = docs[idx : idx + args.bs]
    docs_texts = [e[1] for e in batch]
    doc_ids.extend([e[0] for e in batch])
    docs_inps = tokenizer(
        docs_texts, truncation=True, padding=True, return_tensors="pt"
    ).to(args.device)
    with torch.cuda.amp.autocast(), torch.no_grad():
        batch_embs = model.encode(**docs_inps).to("cpu")
        docs_embs.append(batch_embs)

index = Faiss(d=docs_embs[0].size(1))
docs_embs = torch.cat(docs_embs, dim=0).numpy().astype("float32")
index.add(docs_embs)
# ?for batch_embds in tqdm(docs_embs, desc="Indexing document embeddings"):
# index.add(batch_embs.numpy().astype("float32"))

run = defaultdict(list)
queries_embs = []
for idx in tqdm(
    range(0, len(queries), args.bs),
    desc="Encoding queries and search",
    position=0,
    leave=True,
):
    batch = queries[idx : idx + args.bs]
    query_texts = [e[1] for e in batch]
    query_inps = tokenizer(
        query_texts, truncation=True, padding=True, return_tensors="pt"
    ).to(args.device)
    with torch.cuda.amp.autocast(), torch.no_grad():
        batch_query_embs = (
            model.encode(**query_inps).to("cpu").numpy().astype("float32")
        )
    scores, docs_idx = index.search(batch_query_embs, 1000)
    for idx in range(len(batch)):
        query_id = batch[idx][0]
        for i, score in zip(docs_idx[idx], scores[idx]):
            if i < 0:
                continue
            doc_id = doc_ids[i]
            run[query_id].append((doc_id, score))

write_trec_run(run, args.o)
