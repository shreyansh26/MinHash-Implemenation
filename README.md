# MinHash Implementation

A simple MinHash implementation to identify similar documents based on keywords. A good explanation can be found in the Mining of Massive Datasets course by Stanford. [Chapter 3 till Section 3.3](http://infolab.stanford.edu/~ullman/mmds/ch3.pdf) covers MinHashing and all the concepts required to understand the code.

For large number of documents (10000) in this case, MinHashing is correctly able to identify all 80 pairs of plagiarized documents correctly.

## Overview of steps involved
1. Parse ground truth data to create plagiarized document mappings
2. Converting documents to 3-word shingles and create mapping
3. Defining similarity matrices. Use triangular matrices to reduce memory complexity
4. Creating MinHash signatures for each document
5. Comparing all signatures