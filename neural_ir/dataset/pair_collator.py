class CrossEncoderPairCollator:
    """
    This class prepares inputs for the CrossEncoder by tokenizing and batching a list of sampled pairs.
    Attributes
    ----------
    tokenizer: tokenizers.Tokenizer
        an insstance of HuggingFace's tokenizer 
    max_length: int
        The maximum length of a (query + document) pair. Tokens beyond this limit will be truncated. 
    """

    def __init__(self, tokenizer, max_length):
        """
        Constructing CrossEncoderPairCollator
        Parameters
        ----------
        tokenizer: tokenizers.Tokenizer
        max_length: int 
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        """
        Tokenize and batch a list of (query, document) pairs for CrossEncoder. 
        The output will be used in the method 'neural_ir.trainer.hf_trainer.evaluate'
        Parameters
        ----------
        batch: list
            a list of sampled pairs. Each item of the list is a tuple: batch[i] = (qid, did, query_text, doc_text)
        Returns
        -------
        dict:
            a dictionary formated as {"query_ids": query_ids, "doc_ids": docs_ids, "pairs": pairs} where:
            - query_ids: a list of query ids in the sampled batch
            - doc_ids: a list of docs_ids in the sampled batch
            - pairs: pairs of <query, document> tokenized by the HuggingFace Tokenizer. Should be a dict or an instance of transformers.BatchEncoding.
            For Cross-Encoder, each <query, document> pair must be jointly encoded to form a single input
        """
        pairs = []
        query_ids = []
        doc_ids = []
        for qid, did, query, doc in batch:
            pairs.append((query, doc))
            query_ids.append(qid)
            doc_ids.append(did)
        pairs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {"query_ids": query_ids, "doc_ids": doc_ids, "pairs": pairs}


class BiEncoderPairCollator:
    """
    Prepare inputs for BiEncoder (DenseBiEncoder, SparseBiEncoder).
    Unlike CrossEncoder, queries and documents in BiEncoder should be processed separately 
    Attributes
    ----------
    tokenizer: tokenizers.Tokenizer
        an insstance of HuggingFace's tokenizer 
    query_max_length: int
        maximum length of a query. Any token beyond this limit will be truncated
    doc_max_length: int
        maximum length of a document. Any token beyond this limit will be truncated
    """

    def __init__(self, tokenizer, query_max_length, doc_max_length):
        self.tokenizer = tokenizer
        self.query_max_length = query_max_length
        self.doc_max_length = doc_max_length

    def __call__(self, batch):
        """
        Tokenize and batch a list of (query, document) pairs for BiEncoder 
        The output will be used in the method 'neural_ir.trainer.hf_trainer.evaluate'
        Parameters
        ----------
        batch: list
            a list of sampled pairs. Each item of the list is a tuple: batch[i] = (qid, did, query_text, doc_text)
        Returns
        -------
        dict:
            a dictionary formated as {"query_ids": query_ids, "doc_ids": docs_ids, "queries": queries, "docs": docs} where:
            - query_ids: a list of query ids in the sampled batch
            - doc_ids: a list of docs_ids in the sampled batch
            - queries: all queries in the batch tokenized by the HuggingFace tokenizer. Must be a dict or an instance of transformers.BatchEncoding 
            - docs: all documents in the batch tokenized by the HuggingFace tokenizer. Must be a dict or an instance of transformers.BatchEncoding  
            For Cross-Encoder, each <query, document> pair must be jointly encoded to form a single input
        """
        queries = []
        docs = []
        query_ids = []
        doc_ids = []
        for qid, did, query, doc in batch:
            query_ids.append(qid)
            doc_ids.append(did)
            queries.append(query)
            docs.append(doc)
        queries = self.tokenizer(
            queries,
            padding=True,
            truncation=True,
            max_length=self.query_max_length,
            return_tensors="pt",
        )
        docs = self.tokenizer(
            docs,
            padding=True,
            truncation=True,
            max_length=self.doc_max_length,
            return_tensors="pt",
        )
        return {
            "query_ids": query_ids,
            "doc_ids": doc_ids,
            "queries": queries,
            "docs": docs,
        }

