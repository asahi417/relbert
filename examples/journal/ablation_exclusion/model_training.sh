TEMPLATE="Today, I finally discovered the relation between <subj> and <obj> : <subj> is the <mask> of <obj>"
LM="roberta-base"

#	Part-Whole
REL_ID="part_whole"
MODEL_CKPT="relbert_output/ckpt/exclusion_${REL_ID}"
relbert-train -m "${LM}" -p -a -o "${MODEL_CKPT}" -b 32 -e 10 --loss nce -r 0.000005 -t "${TEMPLATE}" --exclude-relation "2" "2/a" "2/b" "2/c" "2/d" "2/e" "2/f" "2/g" "2/h" "2/i" "2/j"
for E in 1 2 3 4 5 6 7 8 9
do
  relbert-eval-analogy -d 'semeval2012_relational_similarity' -s 'validation' -m "${MODEL_CKPT}/epoch_${E}" -o "${MODEL_CKPT}/epoch_${E}/analogy.forward.json" -b 64
done
relbert-eval-analogy -d 'semeval2012_relational_similarity' -s 'validation' -m "${MODEL_CKPT}/model" -o "${MODEL_CKPT}/model/analogy.forward.json" -b 64

#	Class Inclusion
REL_ID="class_inclusion"
MODEL_CKPT="relbert_output/ckpt/exclusion_${REL_ID}"
relbert-train -m "${LM}" -p -a -o "${MODEL_CKPT}" -b 32 -e 10 --loss nce -r 0.000005 -t "${TEMPLATE}" --exclude-relation "1" "1/a" "1/b" "1/c" "1/d" "1/e"
for E in 1 2 3 4 5 6 7 8 9
do
  relbert-eval-analogy -d 'semeval2012_relational_similarity' -s 'validation' -m "${MODEL_CKPT}/epoch_${E}" -o "${MODEL_CKPT}/epoch_${E}/analogy.forward.json" -b 64
done
relbert-eval-analogy -d 'semeval2012_relational_similarity' -s 'validation' -m "${MODEL_CKPT}/model" -o "${MODEL_CKPT}/model/analogy.forward.json" -b 64

#	Attribute
REL_ID="attribute"
MODEL_CKPT="relbert_output/ckpt/exclusion_${REL_ID}"
relbert-train -m "${LM}" -p -a -o "${MODEL_CKPT}" -b 32 -e 10 --loss nce -r 0.000005 -t "${TEMPLATE}" --exclude-relation "5" "5/a" "5/b" "5/c" "5/d" "5/e" "5/f" "5/g" "5/h" "5/i"
for E in 1 2 3 4 5 6 7 8 9
do
  relbert-eval-analogy -d 'semeval2012_relational_similarity' -s 'validation' -m "${MODEL_CKPT}/epoch_${E}" -o "${MODEL_CKPT}/epoch_${E}/analogy.forward.json" -b 64
done
relbert-eval-analogy -d 'semeval2012_relational_similarity' -s 'validation' -m "${MODEL_CKPT}/model" -o "${MODEL_CKPT}/model/analogy.forward.json" -b 64

#	Contrast
REL_ID="contrast"
MODEL_CKPT="relbert_output/ckpt/exclusion_${REL_ID}"
relbert-train -m "${LM}" -p -a -o "${MODEL_CKPT}" -b 32 -e 10 --loss nce -r 0.000005 -t "${TEMPLATE}" --exclude-relation "4" "4/a" "4/b" "4/c" "4/d" "4/e" "4/f" "4/g" "4/h"
for E in 1 2 3 4 5 6 7 8 9
do
  relbert-eval-analogy -d 'semeval2012_relational_similarity' -s 'validation' -m "${MODEL_CKPT}/epoch_${E}" -o "${MODEL_CKPT}/epoch_${E}/analogy.forward.json" -b 64
done
relbert-eval-analogy -d 'semeval2012_relational_similarity' -s 'validation' -m "${MODEL_CKPT}/model" -o "${MODEL_CKPT}/model/analogy.forward.json" -b 64

#	Similar
REL_ID="similar"
MODEL_CKPT="relbert_output/ckpt/exclusion_${REL_ID}"
relbert-train -m "${LM}" -p -a -o "${MODEL_CKPT}" -b 32 -e 10 --loss nce -r 0.000005 -t "${TEMPLATE}" --exclude-relation "3" "3/a" "3/b" "3/c" "3/d" "3/e" "3/f" "3/g" "3/h"
for E in 1 2 3 4 5 6 7 8 9
do
  relbert-eval-analogy -d 'semeval2012_relational_similarity' -s 'validation' -m "${MODEL_CKPT}/epoch_${E}" -o "${MODEL_CKPT}/epoch_${E}/analogy.forward.json" -b 64
done
relbert-eval-analogy -d 'semeval2012_relational_similarity' -s 'validation' -m "${MODEL_CKPT}/model" -o "${MODEL_CKPT}/model/analogy.forward.json" -b 64