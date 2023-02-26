from torch import nn
import torch
from transformers import AutoModelForMaskedLM


class L1Regularizer(nn.Module):
    def __init__(self, T: int = 5000, alpha: float = 0.01):
        """
        Parameters
        ---------
        T: int 
            number of warming up steps
        alpha: float 
            regularization weight
        """
        super().__init__()
        self.T = T
        self.max_alpha = alpha
        self.current_step = 0
        self.current_alpha = 0

    # TODO: implement this
    def forward(self, reps):
        """
        Calculate L1 for an input tensor then perform a warming up step for the regularization weight
        Parameters
        ----------
        reps: torch.Tensor
            Two dimensional input Tensor
        Returns
        -------
        torch.Tensor:
            The result of L1 applied in the input tensor. 
            L1(reps) = current_alpha * mean(L1(reps_i)) where reps_i is the i-th row of the input
        """
        # BEGIN SOLUTION
        #
        # END SOLUTION

    def step(self):
        """
        Perform a warming up step. This warming up step would allow us to apply the regularization gradually,
        step-by-step with increasing weight over time. Without this warming up, the loss might be overwhelmed by
        the regularization right from the start, leading to the issue that the model produces very sparse but
        not semantically useful vectors in the end.
        Parameters
        ----------
        This method does not have any input parameters
        Returns
        -------
        This method does not return anything
        """
        if self.current_step < self.T:
            self.current_step += 1
            self.current_alpha = (self.current_step / self.T) ** 2 * self.max_alpha
        else:
            pass


class SparseBiEncoder(nn.Module):
    """
    Sparse encoder based on transformer-based masked language model (MLM). 
    Attributes
    ----------
        model: 
            a masked language model resulted from AutoModelFormaskedLM.from_pretrained()
        loss: nn.CrossEntropyLoss
            cross entropy loss for training 
        q_regularizer: L1Regularizer 
            regularizer for queries. This control the sparsity of the query representations
        d_regularizer: L1regularizer 
            regularizer for documents. This control the sparsity of the document representations
    """

    def __init__(self, model_name_or_dir, q_alpha=0.01, d_alpha=0.0001, T=5000):
        super().__init__()
        self.model = AutoModelForMaskedLM.from_pretrained(model_name_or_dir)
        self.loss = nn.CrossEntropyLoss()
        self.q_regularizer = L1Regularizer(alpha=q_alpha, T=T)
        self.d_regularizer = L1Regularizer(alpha=d_alpha, T=T)

    # TODO: implement this method
    def encode(self, input_ids, attention_mask, **kwargs):
        """
        For the Sparse Bi-Encoder, we encode a query/document into a |V|-dimensional vector, where |V| is the size of the vocabulary. 
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
            a two-dimensional tensor whose rows are sparse vectors.
            The output dimension should be batch_size x vocab_size.  Each column represents a vocabulary term. 
            Suppose we have a logit matrix returned by the masked language model,
            you need to perform the following steps to produce the correct output:
            1. Zero-out the logits of all padded tokens (hint: attention_mask parameter)
            2. Apply relu and (natural) log transformation: log(1 + relu(logits))
            3. Return the value of max pooling over the second dimension
        """
        # BEGIN SOLUTION

        # END SOLUTION

    # TODO: implement this method
    def score_pairs(self, queries, docs, return_regularizer=False):
        """
        Retrun scores for pairs of <query, document>
        Since this is a Bi-Encoder, we follow a similar procedure as the Dense Bi-Encoder, and use the dot product to calculate the score for a given query-document pair.
        Parameters
        ----------
        queries: dict or transformers.BatchEncoding
            a batch of queries tokenized by a HuggingFace's tokenizer
        docs: dict or transformers.BatchEncoding
            a batch of docs tokenized by a HuggingFace's tokenizer.  
        return_regularizer: bool
            if True, return the regularizer's output for queries and documents
        Returns
        -------
        if return_regularizer is True:
            return a tuple of (scores, query regularization, document regularization)
        else:
            return scores, where scores[i] = dot(q_vectors[i], d_vectors[i])
        """
        # BEGIN SOLUTION

        # END SOLUTION

    # TODO: implement this method
    def forward(self, queries, pos_docs, neg_docs):
        """
        Given a batch of triplets,  return the loss for training. 
        As in the other two models, we use a contrastive loss with positive and negative pairs. However, we also need to add a regularization term to the loss, which is the sum of the L1 norms of the query and document vectors (more explicitly, we add the norm of the query vector and the mean of the norms of the positive and negative documents). Ultimately, the loss is the sum of the contrastive loss, which is acquired similarly to the other two models, and the regularization term, as previously described.
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
        A tuple of (loss, pos_scores, neg_scores) which are the value of the loss, the estimated score of
        (query, positive document) pairs and the estimated score of (query, negative document) pairs.
        The loss must include the regularization as follows:
        loss = entropy_loss + query_regularization + (positive_regularization + negative_regularization)/2
        """
        # BEGIN SOLUTION

        # END SOLUTION

    def save_pretrained(self, model_name_or_dir, state_dict=None):
        self.model.save_pretrained(model_name_or_dir, state_dict=state_dict)

    @classmethod
    def from_pretrained(cls, model_name_or_dir):
        return cls(model_name_or_dir)

