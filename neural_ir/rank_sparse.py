import argparse
from neural_ir.models.sparse_encoder import SparseBiEncoder
from neural_ir.utils.dataset_utils import read_pairs
from transformers import AutoTokenizer
from tqdm import tqdm
import torch
from collections import defaultdict
import json
from pathlib import Path
import subprocess

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
    default="output/sparse/model",
    type=str,
    help="path to model checkpoint",
)
parser.add_argument(
    "--o",
    type=str,
    default="output/sparse/test_run.trec",
    help="path to output run file",
)
args = parser.parse_args()

docs = read_pairs(args.c)
queries = read_pairs(args.q)

tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
vocabulary = tokenizer.vocab
model = SparseBiEncoder.from_pretrained(args.checkpoint).to(args.device)
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
        docs_embs.append(batch_embs * 100)

docs_embs = torch.cat(docs_embs, dim=0).to(torch.int32)
json_docs = defaultdict(dict)
for row, col in tqdm(
    docs_embs.nonzero(), desc="Converting dense format to sparse format"
):
    doc_id = doc_ids[row]
    term = tokenizer.convert_ids_to_tokens(col.item())
    weight = docs_embs[row, col].item()
    json_docs[doc_id][term] = weight


d_path = Path("output/sparse/docs/")
if not d_path.exists():
    d_path.mkdir(parents=True, exist_ok=True)

with open(d_path / "docs.jsonl", "w") as f:
    for doc_id in tqdm(
        json_docs,
        desc="Writing documents to output/sparse/docs in json format",
        position=0,
        leave=True,
    ):
        f.write(json.dumps({"id": doc_id, "vector": json_docs[doc_id]}) + "\n")

query_embs = []
query_ids = []
for idx in tqdm(
    range(0, len(queries), args.bs),
    desc="Encoding queries and search",
    position=0,
    leave=True,
):
    batch = queries[idx : idx + args.bs]
    query_texts = [e[1] for e in batch]
    query_ids.extend([e[0] for e in batch])
    query_inps = tokenizer(
        query_texts, truncation=True, padding=True, return_tensors="pt"
    ).to(args.device)
    with torch.cuda.amp.autocast(), torch.no_grad():
        batch_query_embs = model.encode(**query_inps).to("cpu")
    query_embs.append(batch_query_embs * 100)
query_embs = torch.cat(query_embs, dim=0).to(torch.int32)
repeat_queries = defaultdict(lambda: "")
for row, col in query_embs.nonzero():
    query_id = query_ids[row]
    term = tokenizer.convert_ids_to_tokens(col.item())
    weight = query_embs[row, col]
    repeat_queries[query_id] = (
        repeat_queries[query_id] + " " + " ".join([term] * weight)
    )
q_path = Path("output/sparse/queries/")
if not q_path.exists():
    q_path.mkdir(parents=True, exist_ok=True)
with open(q_path / "test.tsv", "w") as f:
    for qid in tqdm(repeat_queries, desc=f"Writing queries to {q_path}"):
        f.write(f"{qid}\t{repeat_queries[qid]}\n")

# python -m pyserini.index.lucene   --collection JsonVectorCollection   --input output/sparse/docs   --index test_index   --generator DefaultLuceneDocumentGenerator   --threads 12   --impact --pretokenized
INDEX_COMMAND = """python -m pyserini.index.lucene
    --collection JsonVectorCollection
    --input output/sparse/docs
    --index output/sparse/index 
    --generator DefaultLuceneDocumentGenerator
    --threads 12   --impact --pretokenized"""

RETRIEVE_COMMAND = f"""python -m pyserini.search.lucene
    --index output/sparse/index 
    --topics output/sparse/queries/test.tsv 
    --output {args.o}   
    --output-format trec   
    --batch 36 
    --threads 12  
    --hits 1000  
    --impact
"""

process = subprocess.run(INDEX_COMMAND.split())
process = subprocess.run(RETRIEVE_COMMAND.split())
