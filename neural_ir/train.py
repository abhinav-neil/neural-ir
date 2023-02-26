import argparse
from transformers import AutoTokenizer, TrainingArguments
from neural_ir.dataset import (
    TripletDataset,
    PairDataset,
    CrossEncoderTripletCollator,
    CrossEncoderPairCollator,
)
from pathlib import Path
from neural_ir.dataset.pair_collator import BiEncoderPairCollator
from neural_ir.dataset.triplet_collator import BiEncoderTripletCollator
from neural_ir.models import CrossEncoder
from neural_ir.models.dense_encoder import DenseBiEncoder
from neural_ir.models.sparse_encoder import SparseBiEncoder
from neural_ir.trainer import HFTrainer

parser = argparse.ArgumentParser(description="Training Neural IR models")

parser.add_argument(
    "--pretrained",
    type=str,
    default="distilbert-base-uncased",
    help="Pretrained checkpoint for the base model",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="output",
    help="Output directory to store models and results after training",
)
parser.add_argument(
    "--epochs", type=float, default=1, help="Number of training epochs",
)
parser.add_argument(
    "--train_batch_size", type=int, default=8, help="Training batch size"
)
parser.add_argument(
    "--eval_batch_size", type=int, default=16, help="Evaluation batch size"
)
parser.add_argument(
    "--warmup_steps", type=int, default=5000, help="Evaluation batch size"
)
parser.add_argument(
    "--max_steps", type=int, default=4000, help="Number of training steps"
)
parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate for training")
sub_parsers = parser.add_subparsers(help="Selecting type of models", dest="model")
######################################################################
#           Cross Encoder argument parser                            #
######################################################################
ce_parser = sub_parsers.add_parser("ce", help="Cross Encoder")
ce_parser.add_argument(
    "--max_length", type=int, default=512, help="Maximum lenght of a query + doc pair"
)
######################################################################
#              Dense Encoder argument parser                         #
######################################################################
dense_parser = sub_parsers.add_parser("dense", help="Dense Model")
dense_parser.add_argument(
    "--query_max_length", type=int, default=100, help="Maximum query length"
)
dense_parser.add_argument(
    "--doc_max_length", type=int, default=250, help="Maximum document length"
)
######################################################################
#              Sparse Encoder argument parser                         #
######################################################################
sparse_parser = sub_parsers.add_parser("sparse", help="Sparse Model")
sparse_parser.add_argument(
    "--query_max_length", type=int, default=100, help="Maximum query length"
)
sparse_parser.add_argument(
    "--doc_max_length", type=int, default=250, help="Maximum document length"
)
sparse_parser.add_argument(
    "--q_reg", type=float, default=0.01, help="Query sparse regularization weight"
)
sparse_parser.add_argument(
    "--d_reg", type=float, default=0.0001, help="Document sparse regularization weight"
)
sparse_parser.add_argument(
    "--T", type=int, default=5000, help="Number of warming up steps for regularization"
)
args = parser.parse_args()


OUTPUT_DIR = Path(args.output_dir) / args.model / "model"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

######################################################################
#              Loading a HuggingFace's Tokenizer                     #
######################################################################
tokenizer = AutoTokenizer.from_pretrained(args.pretrained)

######################################################################
#              Loading training and development dataset              #
######################################################################


train_dataset = TripletDataset(
    collection_path="data/collection.tsv",
    queries_path="data/train_queries.tsv",
    train_triplets_path="data/train_triplets.tsv",
)
dev_dataset = PairDataset(
    collection_path="data/collection.tsv",
    queries_path="data/dev_queries.tsv",
    query_doc_pair_path="data/dev_bm25.trec",
    qrels_path="data/dev_qrels.json",
)

######################################################################
#              Instantiating model and data collatators              #
######################################################################

if args.model == "ce":
    triplet_collator = CrossEncoderTripletCollator(tokenizer, args.max_length)
    pair_collator = CrossEncoderPairCollator(tokenizer, args.max_length)
    model = CrossEncoder(args.pretrained)
elif args.model == "dense":
    triplet_collator = BiEncoderTripletCollator(
        tokenizer, args.query_max_length, args.doc_max_length
    )
    pair_collator = BiEncoderPairCollator(
        tokenizer, args.query_max_length, args.doc_max_length
    )
    model = DenseBiEncoder(args.pretrained)
elif args.model == "sparse":
    triplet_collator = BiEncoderTripletCollator(
        tokenizer, args.query_max_length, args.doc_max_length
    )
    pair_collator = BiEncoderPairCollator(
        tokenizer, args.query_max_length, args.doc_max_length
    )
    model = SparseBiEncoder(
        args.pretrained, q_alpha=args.q_reg, d_alpha=args.d_reg, T=args.T
    )
else:
    raise Exception(
        "Invalid selection. Select either {ce, dense, sparse} model for training."
    )

######################################################################
#              TrainingArguments for HuggingFace's Trainer           #
######################################################################
# You can add/customize there training arguments.
# (See: https://huggingface.co/docs/transformers/v4.20.1/en/main_classes/trainer#transformers.TrainingArguments)
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=args.lr,
    num_train_epochs=args.epochs,
    evaluation_strategy="steps",
    fp16=True,
    warmup_steps=args.warmup_steps,
    metric_for_best_model="RR@10",
    load_best_model_at_end=True,
    per_device_train_batch_size=args.train_batch_size,
    per_device_eval_batch_size=args.eval_batch_size,
    max_steps=args.max_steps,
    save_steps=100,
    eval_steps=100,
    save_total_limit=2,
)
######################################################################
#              Instantiating Trainer                                 #
######################################################################
trainer = HFTrainer(
    model,
    train_dataset=train_dataset,
    data_collator=triplet_collator,
    args=training_args,
    eval_dataset=dev_dataset,
    eval_collator=pair_collator,
)
######################################################################
#              Start training                                        #
######################################################################
trainer.train()
trainer.save_model()
