
# Triplet Loss
relbert-train -o relbert_output/ckpt/triplet/batch39 -b 39 -e 1
relbert-eval-analogy -m relbert_output/ckpt/triplet/batch39/model -o relbert_output/ckpt/triplet/batch39/model/analogy.json
relbert-train -o relbert_output/ckpt/triplet/batch39c -b 39 -e 1 -c
relbert-eval-analogy -m relbert_output/ckpt/triplet/batch39c/model -o relbert_output/ckpt/triplet/batch39c/model/analogy.json

# NCE Loss
MODEL="relbert_output/ckpt/nce/batch32nce"
relbert-train -o ${MODEL} -b 32 -e 10 --loss nce --temperature 0.05 -r 0.000005 --num-negative 400
for e in 1 2 3 4 5 6 7 8 9; do
    relbert-eval-analogy -m "${MODEL}/epoch_${e}" -o "${MODEL}/epoch_${e}/analogy.json" -d sat_full
done
relbert-eval-analogy -m "${MODEL}/model" -o "${MODEL}/model/analogy.json" -d sat_full

# ILOOB Loss
MODEL="relbert_output/ckpt/iloob/batch32iloob"
relbert-train -o ${MODEL} -b 32 -e 10 --loss iloob --temperature 0.05 -r 0.000005 --num-negative 400
for e in 1 2 3 4 5 6 7 8 9; do
    relbert-eval-analogy -m "${MODEL}/epoch_${e}" -o "${MODEL}/epoch_${e}/analogy.json" -d sat_full
done
relbert-eval-analogy -m "${MODEL}/model" -o "${MODEL}/model/analogy.json" -d sat_full

"a" "Today, I finally discovered the relation between <subj> and <obj> : <subj> is the <mask> of <obj>"
"b" "Today, I finally discovered the relation between <subj> and <obj> : <obj>  is <subj>'s <mask>"
"c" "Today, I finally discovered the relation between <subj> and <obj> : <mask>"
"d" "I wasn’t aware of this relationship, but I just read in the encyclopedia that <subj> is the <mask> of <obj>"
"e" "I wasn’t aware of this relationship, but I just read in the encyclopedia that <obj>  is <subj>’s <mask>"
