vis () {
  MODEL=${1}
  git clone "https://huggingface.co/relbert/${MODEL}"
  relbert-2d-plot -m "relbert/${MODEL}" -d "relbert/conceptnet_relational_similarity" -o "${MODEL}/vis_conceptnet_relational_similarity"
#  relbert-2d-plot -m "relbert/${MODEL}" -d "relbert/t_rex_relational_similarity" -n "filter_unified.min_entity_4_max_predicate_10" -o "${MODEL}/vis_t_rex_relational_similarity"
  relbert-2d-plot -m "relbert/${MODEL}" -d "relbert/nell_relational_similarity" -o "${MODEL}/vis_nell_relational_similarity"
  cd "${MODEL}" && git add . && git commit -m "Add visualization" && git push && cd ..
  rm -rf "${MODEL}"
}


vis "relbert-roberta-large-nce-semeval2012-0-400"
vis "relbert-roberta-base-nce-semeval2012-0"