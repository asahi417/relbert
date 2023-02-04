

vis () {
  MODEL=${1}
  git clone "https://huggingface.co/relbert/${MODEL}"
  relbert-2d-plot -m "relbert/${MODEL}" -d "relbert/conceptnet_relational_similarity" -o "${MODEL}/vis_conceptnet_relational_similarity"
  relbert-2d-plot -m "relbert/${MODEL}" -d "relbert/t_rex_relational_similarity" -n "filter_unified.min_entity_4_max_predicate_10" -o "${MODEL}/vis_t_rex_relational_similarity"
  relbert-2d-plot -m "relbert/${MODEL}" -d "relbert/nell_relational_similarity" -o "${MODEL}/vis_nell_relational_similarity"
  cd "${MODEL}" && git add . && git commit -m "Add visualization" && git push && cd ..
  rm -rf "${MODEL}"
}


vis "relbert-roberta-large-nce-a-t-rex"
vis "relbert-roberta-large-nce-b-t-rex"
vis "relbert-roberta-large-nce-c-t-rex"
vis "relbert-roberta-large-nce-d-t-rex"
vis "relbert-roberta-large-nce-e-t-rex"


