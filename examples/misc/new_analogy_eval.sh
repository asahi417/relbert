# When there is new dataset added to analogy question, run this to update the metric file of each relbert model

evaluation () {
  MODEL=${1}
  git clone "https://huggingface.co/relbert/${MODEL}"
  # for evaluation
  relbert-eval-analogy -d 'scan' -s 'test' -m "${MODEL}" -o "${MODEL}/analogy.forward.json" -b 64
  relbert-eval-analogy -d 'scan' -s 'test' -m "${MODEL}" -o "${MODEL}/analogy.reverse.json" -b 64 --reverse-pair
  relbert-eval-analogy -d 'scan' -s 'test'  -m "${MODEL}" -o "${MODEL}/analogy.bidirection.json" -b 64 --bi-direction-pair
  relbert-eval-analogy -d 'scan' -s 'validation' -m "${MODEL}" -o "${MODEL}/analogy.forward.json" -b 64
  relbert-eval-analogy -d 'scan' -s 'validation' -m "${MODEL}" -o "${MODEL}/analogy.reverse.json" -b 64 --reverse-pair
  relbert-eval-analogy -d 'scan' -s 'validation' -m "${MODEL}" -o "${MODEL}/analogy.bidirection.json" -b 64 --bi-direction-pair

  # upload
  cd "${MODEL}" && git add . && git commit -m "update metric" && git push && cd ..
  rm -rf "${MODEL}"
}

for loss in "triplet" "iloob"
do
  for prompt in "a" "b" "c" "d" "e"
  do
    evaluation "relbert-roberta-large-${loss}-${prompt}-semeval2012"
  done
done


for data in "semeval2012" "conceptnet" "nell" "t-rex" "semeval2012-nell" "semeval2012-t-rex" "semeval2012-nell-t-rex"
do
  for prompt in "a" "b" "c" "d" "e"
  do
    evaluation "relbert-roberta-large-nce-${prompt}-${data}"
  done
done
