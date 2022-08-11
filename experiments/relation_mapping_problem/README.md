# Relation Mapping Problem
Relation mapping `M` is the set of bijective map in between two sets of terms (`A` and `B`):
```
[set `A`]: ("solar system", "sun", "planet", "mass", "attracts", "revolves", "gravity")
[set `B`]: ("atom", "nucleus", "electron", "charge", "attracts", "revolves", "electromagnetism")

[Relation Mapping `M`]
* "solar system"   -> "atom"
* "sun"            -> "nucleus"
* "planet"         -> "electron"
* "mass"           -> "charge"
* "attracts"       -> "attracts"
* "revolves"       -> "revolves"
* "gravity"        -> "electromagnetism"
```

***[Relation Mapping Problem](https://www.jair.org/index.php/jair/article/view/10583)*** is the task to identify the mapping `M` given the sets of terms `A` and `B`.

## Dataset
The [dataset file](./data.jsonl) is a jsonline where each line is a json data containing following data.

- `source`: A list of terms, which is the source of the relation mapping from.
- `target_random`: A list of terms, where we want to find a mapping from `source` to.
- `target`: A correctly ordered `target_random` that aligns with the `source`. 

Given `source` and `target_random`, the task is to predict the correct order of `target_random` so that it matches `target`.
In average 7 terms are in the set, so the total number of possible order is 5040.
  

## Approach
As an approach to solve the relation mapping with RelBERT (or relation embedding model in general), we can follow something like this.
In a permutation `P:=[(a_1, b_1), ..., (a_7, b_7)]`, we compute a relation embedding of each word pair `(a_i, b_i)`
and intuitively the permutation is valid if all the word pairs hold same relation, meaning their relation embeddings are
close each other. So we can somehow calculate coherence of the relation embeddings and choose the most coherent permutation. 
