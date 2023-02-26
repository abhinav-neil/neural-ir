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
|     DenseEncoder     |  x         |   x         |   
|     SparseEncoder    |  x         |   x         |  
|     CrossEncoder     |  x         |   x         |
|     Your Creativity  |  x         |   x         |  

# Description of Your Creativity (fill this in; max 200 words)
> In this section, describe the new method variant you have implemented. A reader who is familiar with the above models should be able to read this description and understand exactly what you have changed

# Your summary (fill this in; max 400 words):
> :memo:
