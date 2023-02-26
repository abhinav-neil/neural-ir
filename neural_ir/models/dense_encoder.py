from torch import nn
import torch
from transformers import AutoModel


class DenseBiEncoder(nn.Module):
    """
    Unlike the CrossEncoder, the DenseBiEncoder encodes queries and documents separately into two dense vectors.
    The score between a query and document is calculated as the dot product between their dense representations.
    The document representation in the DenseBiEncoder can be pre-computed and indexed offline using vector indexing
    toolkits such as Faiss. On the other hand, the query representation needs to be computed on-the-fly because
    we don't know the queries beforehand.
    Attributes
    ----------
    model: result of transformers.AutoModel.from_pretrained()
    loss: nn.CrossEntropyLoss
        Cross entropy loss for training
    """

    def __init__(self, model_name_or_dir) -> None:
        """
        Constructing DenseBiEncoder 
        Parameters
        ----------
        model_name_or_dir: str
            a HuggingFace's model name or path to a model's checkpoint
        """
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name_or_dir)
        self.loss = nn.CrossEntropyLoss()

    # TODO: Implement this method
    def encode(self, input_ids, attention_mask, **kwargs):
        """
        Encode a batch of text inputs to a batch of dense vectors
        The Dense Bi-Encoder uses the final hidden states as the representation of its input. To do so, an aggregation function is used to reduce the sequence of hidden states to a single vector. This is typically done by taking the mean or the max of the hidden states, or by exclusively using the representation of a special token like [CLS]. For this assignment, we will use the mean of the hidden states, excluding the padding tokens
        Parameters
        ----------
        input_ids: torch.Tensor
            token ids returned by a HuggingFace's tokenizer
        attention_mask:
            attention mask returned by a HuggingFace's tokenizer
        **kwargs: 
            other inputs returned by a HuggingFace's tokenizer
        Returns
        -------
        torch.Tensor
            a two-dimensional tensor whose rows are dense vectors, where the
            dense vectors are the mean of the last layer's embeddings of
            all tokens in the sequence except for the padded tokens.
        """
        # BEGIN SOLUTION
        # END SOLUTION

    # TODO: Implement this method
    def score_pairs(self, queries, docs):
        """
        Calculating the scores of query, document pairs. scores[i] = score(queries[i], docs[i])
        As the queries and documents are encoded separately, the score for a given query-document pair can be defined as either the dot product or the cosine similarity between the two vectors. For this assignment, we will use the dot product.
        Parameters
        ----------
        queries: dict or transformers.BatchEncoding
            a batch of queries tokenized by a HuggingFace's tokenizer
        docs: dict or transformers.BatchEncoding
            a batch of docs tokenized by a HuggingFace's tokenizer. 
            queries and docs must contain the same number of items
        Returns
        -------
        torch.Tensor:
            a scores vector where scores[i] = dot(q_vectors[i], d_vectors[i])
        """
        # BEGIN SOLUTION
        # END SOLUTION

    # TODO: Implement this method
    def forward(self, queries, pos_docs, neg_docs):
        """
        Forward method used during training
        Similarly to the Cross-Encoder, we will train the Dense Bi-Encoder with a contrastive loss. As a result, the same procedure should be followed to calculate the loss.
        Parameters
        ----------
        queries: dict or transformers.BatchEncoding
            a batch of queries tokenized by a HuggingFace's tokenizer
        pos_docs: dict or transformers.BatchEncoding
            a batch of positive docs tokenized by a HuggingFace's tokenizer. 
        neg_docs: dict or transformers.BatchEncoding
            a batch of negative docs tokenized by a HuggingFace's tokenizer. 
        queries, pos_docs, neg_docs must contain the same number of items
        Returns
        -------
        A tuple of (loss, pos_scores, neg_scores) which are the value of the (CrossEntropy) loss, the estimated score of
        (query, positive document) pairs and the estimated score of (query, negative document) pairs.
        The loss should pull the positive document closer to the query and push the negative document away.
        """
        # BEGIN SOLUTION
        # END SOLUTION

    def save_pretrained(self, model_dir, state_dict=None):
        """
        Save the model's checkpoint to a directory
        Parameters
        ----------
        model_dir: str or Path
            path to save the model checkpoint to
        """
        self.model.save_pretrained(model_dir, state_dict=state_dict)

    @classmethod
    def from_pretrained(cls, model_name_or_dir):
        """
        Load model checkpoint for a path or directory
        Parameters
        ----------
        model_name_or_dir: str
            a HuggingFace's model or path to a local checkpoint
        """
        return cls(model_name_or_dir)

