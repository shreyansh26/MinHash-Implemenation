import binascii
import time
import logging
import sys
import random

logging.basicConfig(level=logging.INFO)

DEBUG = False
HASH_COMPONENTS = 10
NUM_DOCS = 10000

train_file = f"data/articles_{NUM_DOCS}.train"
ground_truth_file = f"data/articles_{NUM_DOCS}.truth"

# Parse ground truth data to create plagiarized document mappings
logging.info('Parse ground truth data to create plagiarized document mappings')

plagiarized = dict()

fobj = open(ground_truth_file)

for line in fobj.readlines():
    line = line.strip()
    doc_pairs = line.split(' ')

    plagiarized[doc_pairs[0]] = doc_pairs[1]
    plagiarized[doc_pairs[1]] = doc_pairs[0]

if DEBUG:
    print(plagiarized)

fobj.close()

# Convert document to 3-word shingles
logging.info('Converting documents to 3-word shingles and create mapping')

doc_shingle_mapping = dict()
total_shingles = 0
doc_ids = []

start_time = time.time()

fobj = open(train_file)

for line in fobj.readlines():
    line = line.strip()
    line = line.split(" ")

    doc_id = line[0]
    document = line[1:]

    shingles = set()

    for idx in range(len(document)-2):
        shingle = document[idx] + " " + document[idx+1] + " " + document[idx+2]

        # Hash the shingle using CRC32
        shingle_hash = binascii.crc32(shingle.encode('utf8')) & 0xffffffff

        shingles.add(shingle_hash)

    doc_ids.append(doc_id)
    doc_shingle_mapping[doc_id] = shingles
    total_shingles += len(document) - 2

fobj.close()

end_time = time.time()

print(f"Time taken for {NUM_DOCS}: {end_time-start_time:.2f} seconds.")
print(f"Average shingles per doc: {total_shingles/NUM_DOCS:.2f}.")

# Define similarity matrices. Use triangular matrices to reduce memory complexity
logging.info('Defining similarity matrices. Use triangular matrices to reduce memory complexity')

num_comparisons = (NUM_DOCS * (NUM_DOCS-1)) // 2

naive_jaccard_similarity = [0]*num_comparisons
minhash_jaccard_similarity = [0]*num_comparisons

def get_triangle_index(doc_x, doc_y):
    if doc_x == doc_y:
        logging.error('Same document ids')
        sys.exit(0)

    if doc_x > doc_y:
        doc_x, doc_y = doc_y, doc_x

    # Simplified form of - https://stackoverflow.com/questions/27086195/linear-index-upper-triangular-matrix
    ind = int((doc_x * NUM_DOCS) - (doc_x * (doc_x + 1) / 2) + doc_y - doc_x) - 1
    return ind

if DEBUG:
    # Compute naive pair-wise Jaccard similarities
    logging.info('Computing naive pair-wise Jaccard similarities')

    start_time = time.time()

    for doc_x in range(NUM_DOCS):
        shingles1 = doc_shingle_mapping[doc_ids[doc_x]]

        for doc_y in range(doc_x + 1, NUM_DOCS):
            shingles2 = doc_shingle_mapping[doc_ids[doc_y]]

            naive_jaccard_similarity[get_triangle_index(doc_x, doc_y)] = len(shingles1.intersection(shingles2)) / len(shingles1.union(shingles2))

        if doc_x % 100 == 0:
            print(f"{doc_x} of {NUM_DOCS} done.")

    end_time = time.time()

    print(f"Naive Jaccard-similarity calculation took {end_time-start_time:.2f} seconds.")

    del naive_jaccard_similarity

# Create MinHash signatures for each document
logging.info('Creating MinHash signatures for each document')

start_time = time.time()

MAX_SINGLE_HASH = 2**32-1
NEXT_LARGEST_PRIME = 4294967311

# Create hash functions of the form h = (ax + b) % c
# where a, b are random coefficients and c is a prime number
# just greater than the maximum shingle hash

# Generate n random coefficients
def get_random_coefficients(n):
  coefficients_list = []
  
  while n > 0:
    # Get a random shingle ID.
    randIndex = random.randint(0, MAX_SINGLE_HASH) 
  
    # Ensure that each random number is unique.
    while randIndex in coefficients_list:
      randIndex = random.randint(0, MAX_SINGLE_HASH) 
    
    # Add the random number to the list.
    coefficients_list.append(randIndex)
    n = n - 1
    
  return coefficients_list

COEFFICIENTS_A = get_random_coefficients(HASH_COMPONENTS)
COEFFICIENTS_B = get_random_coefficients(HASH_COMPONENTS)

doc_minhash_mapping = dict()

for doc_id in doc_ids:
    shingle_set = doc_shingle_mapping[doc_id]

    signature = []

    for idx in range(HASH_COMPONENTS):
        # Initialize min_hash_code to track the lowest shingle_hash that was seen.
        # For details, refer section 3.3.5 of http://infolab.stanford.edu/~ullman/mmds/ch3.pdf
        min_hash_code = 1000000000000

        for shingle_id in shingle_set:
            # use hash function idx with coefficients a, b calculated earlier
            shingle_hash_id = (COEFFICIENTS_A[idx] * shingle_id + COEFFICIENTS_B[idx]) % NEXT_LARGEST_PRIME

            if shingle_hash_id < min_hash_code:
                min_hash_code = shingle_hash_id

        # Add the min_hash_code as component idx of the signature.
        signature.append(min_hash_code)

    # MinHash signature for current document doc_id
    doc_minhash_mapping[doc_id] = signature

end_time = time.time()

print(f"Generating MinHash signatures took {end_time-start_time:.2f} seconds.")

# Compare all signatures
logging.info('Comparing all signatures')

start_time = time.time()

for doc_x in range(NUM_DOCS):
    minhash_signature1 = doc_minhash_mapping[doc_ids[doc_x]]

    for doc_y in range(doc_x + 1, NUM_DOCS):
        minhash_signature2 = doc_minhash_mapping[doc_ids[doc_y]]

        equal_components = 0
        # Count the number of positions in the minhash signature which are equal.
        for i in range(0, HASH_COMPONENTS):
            equal_components += minhash_signature1[i] == minhash_signature2[i]

        minhash_jaccard_similarity[get_triangle_index(doc_x, doc_y)] = equal_components / HASH_COMPONENTS

    if doc_x % 100 == 0:
        print(f"{doc_x} of {NUM_DOCS} done.")

end_time = time.time()

print(f"MinHash Jaccard-similarity calculation took {end_time-start_time:.2f} seconds.")

# Calculate metrics
logging.info('Calculate document similarity metrics of Naive vs MinHash')

tp, fp = 0, 0
threshold = 0.5

for doc_x in range(NUM_DOCS):
    for doc_y in range(doc_x + 1, NUM_DOCS):
        minhash_jaccard_score = minhash_jaccard_similarity[get_triangle_index(doc_x, doc_y)]

        # If similarity is above the threshold
        if minhash_jaccard_score > threshold:
            shingles1 = doc_shingle_mapping[doc_ids[doc_x]]
            shingles2 = doc_shingle_mapping[doc_ids[doc_y]]
            actual_jaccard_score = len(shingles1.intersection(shingles2)) / len(shingles1.union(shingles2))

            # Print out the match and similarity values with pretty spacing.
            print("  %5s --> %5s   %.2f     %.2f" % (doc_ids[doc_x], doc_ids[doc_y], minhash_jaccard_score, actual_jaccard_score))

            if plagiarized[doc_ids[doc_x]] == doc_ids[doc_y]:
                tp = tp + 1
            else:
                fp = fp + 1

# Display true positive and false positive counts.
print("True positives: " + str(tp) + " / " + str(int(len(plagiarized.keys()) / 2)))
print("False positives: " + str(fp))