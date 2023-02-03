

MODEL="relbert-roberta-large-triplet-d-semeval2012"
git clone "https://huggingface.co/relbert/${MODEL}"
relbert-2d-plot -m "relbert/${MODEL}" -d "relbert/conceptnet_relational_similarity" -o "${MODEL}/vis_conceptnet_relational_similarity"
