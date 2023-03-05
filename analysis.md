# Overview
The analysis portion of this assignment is similar to the analysis in part 1, with the addition of a method variant you come up with.
To receive all 50 points for this portion of the assignment, you need to:
1. Come up with a variant of one of the neural methods considered in the assignment, reporting the results of your variant in the table below and saving its results to `output/your_creativity/test_run.trec`. (Replace the row named _Your Creativity_.) To receive full credit, your variant should outperform the base approach that you modified. For example, if you modify the CrossEncoder base model, your variant needs to outperform the CrossEncoder for full credit.
2. Describe what this variant is and why you believe this variant may lead to improved performance. (Fill in _Description of Your Creativity_.)
3. Conduct an analysis of how the DenseEncoder, SparseEncoder, CrossEncoder, and your variant perform, as you did in part 1. (Fill in _Your Summary_.)

By "method variant", we mean some change to the DenseEncoder, SparseEncoder, or CrossEncoder. For example, thinking back to part 1, a variant of BM25 could modify the BM25 TF formula in some way (e.g., log).

# Results table (fill this in):

| Model                | MRR@10     | R@1000      |
|----------------------|------------|-------------|
|     BM25             |  0.665     |   0.982     |   
|     DenseEncoder     |  0.50        |   0.94        |   
|     SparseEncoder    |  0.72         |   0.99       |  
|     CrossEncoder     |  0.69      |   0.92      |
|     Your Creativity  |  0.90        |   0.44         |  

# Description of Your Creativity (fill this in; max 200 words)
We develop a variant of the dense encoder using cosine similarity instead of dot product similiarity. This is defined as:

$$ \text{sim}(q, d) = \frac{q \cdot d}{\|q\| \|d\|} $$

We believe that this a better similarity metric because it normalizes the dot product by the magnitude of the query and document vectors. This is important because the dot product is sensitive to the magnitude of the vectors, and we want to avoid this sensitivity. This is because the magnitude of the vectors is dependent on the length of the query and document, which is not relevant to the similarity between the query and document. The cosine similarity metric captures the similarity between the query and document without being sensitive to the magnitude of the vectors.

# Your summary (fill this in; max 400 words):

We compared different models for ranking and reranking documents relevant to a query. Of these, BM25 is a statistical model which uses metrics such as term-frequency and inverse document frequency to rank documents. In contrast, the neural models are content-based, and find relevant documents for a query based on the similarity between query and document. This is achieved by building in-context representations of the query and document using a tranformer for encoding. 
We compare 3 different approaches to this. The cross-encoder takes in the query and the document as input, and produces a single output score indicating the relevance of the document to the query. The advantage of this approach is that the model can take into account the interaction between the query and the document while encoding them, which can potentially lead to better performance. However, cross-encoder models can be computationally expensive. We find that the cross-encoder achieves high performance in terms of recall & MRR (mean reciprocal rank) when it is used for reranking. In contrast, the dense retrieval methods encode the query and document separately using a bi-encoder, and then output a similarity score between document & query using a metric like dot product similarity or cosine similarity. We also use a contrastive loss during training so the model learns to maximize similarity between positive examples and minimize similairty between negative examples. This captures less contextual information about the relationship between the 2, compared to the cross-encoder, but is much faster since the document represenations can be stored in advance. In dense retrieval, the bi-encoder builds dense representations of document & query, whereas in learned sparse retrieval, the encoder learns term weights for the document & query such as by using an masked language model (MLM), and then imposes a regularization term (such as $l1$) to the loss to impose sparsity of terms. This results in more efficiency (and less computational time) than dense retrieval, but possibly at the cost slightly lower perforamnce (due to information loss). We observe that both dense and aprse methods perform comparably in our setting, achieving hig recall & MRR. While we use dot product similarity for the original dense & sparse encoder, we also introduce a variant of the dense encoder using cosine similarity, which performs slightly worse than the original one.
