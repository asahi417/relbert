for loss in 'triplet' 'nce' 'iloob'
do
  for prompt in 'a' 'b' 'c' 'd' 'e'
  do
    data="semeval2012"
    git clone "https://huggingface.co/relbert/relbert-roberta-large-${loss}-${prompt}-${data}"
    relbert-eval-analogy --overwrite -d "t_rex_relational_similarity" "conceptnet_relational_similarity" "nell_relational_similarity" -s "test" -m "relbert-roberta-large-${loss}-${prompt}-${data}" -o "relbert-roberta-large-${loss}-${prompt}-${data}/analogy.forward.json" -b 64
    relbert-eval-analogy --overwrite -d "t_rex_relational_similarity" "conceptnet_relational_similarity" "nell_relational_similarity" -s "test" -m "relbert-roberta-large-${loss}-${prompt}-${data}" -o "relbert-roberta-large-${loss}-${prompt}-${data}/analogy.reverse.json" -b 64 --reverse-pair
    relbert-eval-analogy --overwrite -d "t_rex_relational_similarity" "conceptnet_relational_similarity" "nell_relational_similarity" -s "test"  -m "relbert-roberta-large-${loss}-${prompt}-${data}" -o "relbert-roberta-large-${loss}-${prompt}-${data}/analogy.bidirection.json" -b 64 --bi-direction-pair
    relbert-eval-analogy --overwrite -d "t_rex_relational_similarity" "conceptnet_relational_similarity" "nell_relational_similarity" -s "validation" -m "relbert-roberta-large-${loss}-${prompt}-${data}" -o "relbert-roberta-large-${loss}-${prompt}-${data}/analogy.forward.json" -b 64
    relbert-eval-analogy --overwrite -d "t_rex_relational_similarity" "conceptnet_relational_similarity" "nell_relational_similarity" -s "validation" -m "relbert-roberta-large-${loss}-${prompt}-${data}" -o "relbert-roberta-large-${loss}-${prompt}-${data}/analogy.reverse.json" -b 64 --reverse-pair
    relbert-eval-analogy --overwrite -d "t_rex_relational_similarity" "conceptnet_relational_similarity" "nell_relational_similarity" -s "validation" -m "relbert-roberta-large-${loss}-${prompt}-${data}" -o "relbert-roberta-large-${loss}-${prompt}-${data}/analogy.bidirection.json" -b 64 --bi-direction-pair
    relbert-push-to-hub -m "relbert-roberta-large-${loss}-${prompt}-${data}" -a "relbert-roberta-large-${loss}-${prompt}-${data}"
  done
done


loss='nce'
for data in "t-rex" "nell" "conceptnet"
do
  for prompt in 'a' 'b' 'c' 'd' 'e'
  do
    git clone "https://huggingface.co/relbert/relbert-roberta-large-${loss}-${prompt}-${data}"
    relbert-eval-analogy --overwrite -d "t_rex_relational_similarity" "conceptnet_relational_similarity" "nell_relational_similarity" -s "test" -m "relbert-roberta-large-${loss}-${prompt}-${data}" -o "relbert-roberta-large-${loss}-${prompt}-${data}/analogy.forward.json" -b 64
    relbert-eval-analogy --overwrite -d "t_rex_relational_similarity" "conceptnet_relational_similarity" "nell_relational_similarity" -s "test" -m "relbert-roberta-large-${loss}-${prompt}-${data}" -o "relbert-roberta-large-${loss}-${prompt}-${data}/analogy.reverse.json" -b 64 --reverse-pair
    relbert-eval-analogy --overwrite -d "t_rex_relational_similarity" "conceptnet_relational_similarity" "nell_relational_similarity" -s "test"  -m "relbert-roberta-large-${loss}-${prompt}-${data}" -o "relbert-roberta-large-${loss}-${prompt}-${data}/analogy.bidirection.json" -b 64 --bi-direction-pair
    relbert-eval-analogy --overwrite -d "t_rex_relational_similarity" "conceptnet_relational_similarity" "nell_relational_similarity" -s "validation" -m "relbert-roberta-large-${loss}-${prompt}-${data}" -o "relbert-roberta-large-${loss}-${prompt}-${data}/analogy.forward.json" -b 64
    relbert-eval-analogy --overwrite -d "t_rex_relational_similarity" "conceptnet_relational_similarity" "nell_relational_similarity" -s "validation" -m "relbert-roberta-large-${loss}-${prompt}-${data}" -o "relbert-roberta-large-${loss}-${prompt}-${data}/analogy.reverse.json" -b 64 --reverse-pair
    relbert-eval-analogy --overwrite -d "t_rex_relational_similarity" "conceptnet_relational_similarity" "nell_relational_similarity" -s "validation" -m "relbert-roberta-large-${loss}-${prompt}-${data}" -o "relbert-roberta-large-${loss}-${prompt}-${data}/analogy.bidirection.json" -b 64 --bi-direction-pair
    relbert-push-to-hub -m "relbert-roberta-large-${loss}-${prompt}-${data}" -a "relbert-roberta-large-${loss}-${prompt}-${data}"
  done
done

