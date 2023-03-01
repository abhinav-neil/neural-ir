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

d_path = Path("output/sparse/docs/")
if not d_path.exists():
    d_path.mkdir(parents=True, exist_ok=True)

doc_file = open(d_path / "docs.jsonl", "w")


def write_doc_to_file(batch_embds, batch_doc_ids, output_file):
    batch_embds = (batch_embds * 100).to(torch.int32)
    k_values, k_indices = batch_embds.topk(400, dim=1)
    for doc_id, token_ids, token_weights in zip(
        batch_doc_ids, k_indices.tolist(), k_values.tolist()
    ):
        doc_json = {"id": doc_id, "vector": dict(zip(token_ids, token_weights))}
        output_file.write(json.dumps(doc_json) + "\n")


for idx in tqdm(
    range(0, len(docs), args.bs), desc="Encoding documents", position=0, leave=True
):
    batch = docs[idx : idx + args.bs]
    docs_texts = [e[1] for e in batch]
    batch_doc_ids = [e[0] for e in batch]
    docs_inps = tokenizer(
        docs_texts, truncation=True, padding=True, return_tensors="pt"
    ).to(args.device)
    with torch.cuda.amp.autocast(), torch.no_grad():
        batch_embs = model.encode(**docs_inps).to("cpu")
        write_doc_to_file(batch_embs, batch_doc_ids, doc_file)
doc_file.close()

q_path = Path("output/sparse/queries/")
if not q_path.exists():
    q_path.mkdir(parents=True, exist_ok=True)
query_file = open(q_path / "test.tsv", "w")


def write_query_to_file(batch_embds, batch_query_ids, output_file):
    batch_embds = (batch_embds * 100).to(torch.int32)
    # repeat_queries = defaultdict(lambda: "")
    k_values, k_indices = batch_embds.topk(400, dim=1)
    for query_id, token_ids, token_weights in zip(
        batch_query_ids, k_indices.tolist(), k_values.tolist()
    ):
        repeat_query = " ".join(
            [
                " ".join([str(token_id)] * token_weight)
                for token_id, token_weight in zip(token_ids, token_weights)
                if token_weight > 0
            ]
        )
        output_file.write(f"{query_id}\t{repeat_query}\n")
        # doc_json = {"id": doc_id, "vector": dict(zip(token_ids, token_weights))}
        # output_file.write(json.dumps(doc_json) + "\n")
        # for qid in repeat_queries:
        # output_file.write(f"{qid}\t{repeat_queries[qid]}\n")


for idx in tqdm(
    range(0, len(queries), args.bs),
    desc="Encoding queries and search",
    position=0,
    leave=True,
):
    batch = queries[idx : idx + args.bs]
    query_texts = [e[1] for e in batch]
    batch_query_ids = [e[0] for e in batch]
    query_inps = tokenizer(
        query_texts, truncation=True, padding=True, return_tensors="pt"
    ).to(args.device)
    with torch.cuda.amp.autocast(), torch.no_grad():
        batch_query_embs = model.encode(**query_inps).to("cpu")
        write_query_to_file(batch_query_embs, batch_query_ids, query_file)
query_file.close()
# query_embs.append(batch_query_embs * 100)
# query_embs = torch.cat(query_embs, dim=0).to(torch.int32)


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
