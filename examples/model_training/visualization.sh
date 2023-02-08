

vis () {
  MODEL=${1}
  git clone "https://huggingface.co/relbert/${MODEL}"
  relbert-2d-plot -m "relbert/${MODEL}" -d "relbert/conceptnet_relational_similarity" -o "${MODEL}/vis_conceptnet_relational_similarity"
  relbert-2d-plot -m "relbert/${MODEL}" -d "relbert/t_rex_relational_similarity" -n "filter_unified.min_entity_4_max_predicate_10" -o "${MODEL}/vis_t_rex_relational_similarity"
  relbert-2d-plot -m "relbert/${MODEL}" -d "relbert/nell_relational_similarity" -o "${MODEL}/vis_nell_relational_similarity"
  cd "${MODEL}" && git add . && git commit -m "Add visualization" && git push && cd ..
  rm -rf "${MODEL}"
}


vis "relbert-roberta-large-triplet-a-semeval2012"
vis "relbert-roberta-large-triplet-b-semeval2012"
vis "relbert-roberta-large-triplet-c-semeval2012"
vis "relbert-roberta-large-triplet-d-semeval2012"
vis "relbert-roberta-large-triplet-e-semeval2012"


vis "relbert-roberta-large-iloob-a-semeval2012"
vis "relbert-roberta-large-iloob-b-semeval2012"
vis "relbert-roberta-large-iloob-c-semeval2012"
vis "relbert-roberta-large-iloob-d-semeval2012"

[TODO] vis "relbert-roberta-large-iloob-e-semeval2012"


vis "relbert-roberta-large-nce-a-semeval2012"
vis "relbert-roberta-large-nce-b-semeval2012"
vis "relbert-roberta-large-nce-c-semeval2012"
vis "relbert-roberta-large-nce-d-semeval2012"
vis "relbert-roberta-large-nce-e-semeval2012"


vis "relbert-roberta-large-nce-a-t-rex"
vis "relbert-roberta-large-nce-b-t-rex"
vis "relbert-roberta-large-nce-c-t-rex"
vis "relbert-roberta-large-nce-d-t-rex"
vis "relbert-roberta-large-nce-e-t-rex"


vis "relbert-roberta-large-nce-a-conceptnet"
vis "relbert-roberta-large-nce-b-conceptnet"
vis "relbert-roberta-large-nce-c-conceptnet"
vis "relbert-roberta-large-nce-d-conceptnet"
vis "relbert-roberta-large-nce-e-conceptnet"


vis "relbert-roberta-large-nce-a-nell"
vis "relbert-roberta-large-nce-b-nell"
vis "relbert-roberta-large-nce-c-nell"
vis "relbert-roberta-large-nce-d-nell"
vis "relbert-roberta-large-nce-e-nell"

