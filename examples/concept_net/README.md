# ConceptNet Relation Clustering

## Get Started
```shell
pip install hdbscan
```

## Scripts
### Step 1: Process ConceptNet
```shell
python preprocess_concept_net.py
```
This creates a data directory where the processed ConceptNet word pairs are stored.

### Step 2: Compute RelBERT Embedding/Cache as a Gensim Static Embedding
```shell
python get_embedding.py
```
This generates a model file `relbert_embedding.bin` that is a gensim-model file of relation embedding over the word pairs from the processed ConceptNet.

### Step 3: Clustering

### Step 4: 2-D Visualization