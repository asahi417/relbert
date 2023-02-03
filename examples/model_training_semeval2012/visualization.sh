

MODEL="relbert-roberta-large-triplet-d-semeval2012"
git clone "https://huggingface.co/relbert/${MODEL}"
relbert-2d-plot -m "relbert/${MODEL}" -d "relbert/conceptnet_relational_similarity" -o "${MODEL}/vis_conceptnet_relational_similarity"
relbert-2d-plot -m "relbert/${MODEL}" -d "relbert/t_rex_relational_similarity" -o "${MODEL}/vis_nell_relational_similarity"
#relbert-2d-plot -m "relbert/${MODEL}" -d "relbert/nell_relational_similarity" -o "${MODEL}/vis_nell_relational_similarity"
cd "${MODEL}" && git add . && git commit -m "Add visualization" && git push && cd ..