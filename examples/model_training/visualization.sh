

vis () {
  MODEL=${1}
  git clone "https://huggingface.co/relbert/${MODEL}"
  relbert-2d-plot -m "relbert/${MODEL}" -d "relbert/conceptnet_relational_similarity" -o "${MODEL}/vis_conceptnet_relational_similarity"
  relbert-2d-plot -m "relbert/${MODEL}" -d "relbert/t_rex_relational_similarity" -n "filter_unified.min_entity_4_max_predicate_10" -o "${MODEL}/vis_t_rex_relational_similarity"
  relbert-2d-plot -m "relbert/${MODEL}" -d "relbert/nell_relational_similarity" -o "${MODEL}/vis_nell_relational_similarity"
  cd "${MODEL}" && git add . && git commit -m "Add visualization" && git push && cd ..
  rm -rf "${MODEL}"
}

for loss in "triplet" "iloob"
do
  for prompt in "a" "b" "c" "d" "e"
  do
    vis "relbert-roberta-large-${loss}-${prompt}-semeval2012"
  done
done


for data in "semeval2012" "conceptnet" "nell" "t-rex" "semeval2012-nell" "semeval2012-t-rex" "semeval2012-nell-t-rex"
do
  for prompt in "a" "b" "c" "d" "e"
  do
    vis "relbert-roberta-large-nce-${prompt}-${data}"
  done
done

vis "relbert-roberta-base-nce-a-semeval2012"