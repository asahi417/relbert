# Triplet Loss
relbert-train -o relbert_output/ckpt/triplet/batch39 -b 39 -e 1
relbert-eval-analogy -m relbert_output/ckpt/triplet/batch39/model -o relbert_output/ckpt/triplet/batch39/model/analogy.json
relbert-train -o relbert_output/ckpt/triplet/batch39c -b 39 -e 1 -c
relbert-eval-analogy -m relbert_output/ckpt/triplet/batch39c/model -o relbert_output/ckpt/triplet/batch39c/model/analogy.json

# NCE Loss
relbert-train -o relbert_output/ckpt/nce/batch32nce -b 32 -e 10 --loss nce --temperature 0.05 -r 0.000005 --num-negative 400
relbert-eval-analogy -m relbert_output/ckpt/nce/batch32nce/epoch_1 -o relbert_output/ckpt/nce/batch32nce/epoch_1/analogy.json -d sat_full
relbert-eval-analogy -m relbert_output/ckpt/nce/batch32nce/epoch_2 -o relbert_output/ckpt/nce/batch32nce/epoch_2/analogy.json -d sat_full
relbert-eval-analogy -m relbert_output/ckpt/nce/batch32nce/epoch_3 -o relbert_output/ckpt/nce/batch32nce/epoch_3/analogy.json -d sat_full
relbert-eval-analogy -m relbert_output/ckpt/nce/batch32nce/epoch_4 -o relbert_output/ckpt/nce/batch32nce/epoch_4/analogy.json -d sat_full
relbert-eval-analogy -m relbert_output/ckpt/nce/batch32nce/epoch_5 -o relbert_output/ckpt/nce/batch32nce/epoch_5/analogy.json -d sat_full
relbert-eval-analogy -m relbert_output/ckpt/nce/batch32nce/epoch_6 -o relbert_output/ckpt/nce/batch32nce/epoch_6/analogy.json -d sat_full
relbert-eval-analogy -m relbert_output/ckpt/nce/batch32nce/epoch_7 -o relbert_output/ckpt/nce/batch32nce/epoch_7/analogy.json -d sat_full
relbert-eval-analogy -m relbert_output/ckpt/nce/batch32nce/epoch_8 -o relbert_output/ckpt/nce/batch32nce/epoch_8/analogy.json -d sat_full
relbert-eval-analogy -m relbert_output/ckpt/nce/batch32nce/epoch_9 -o relbert_output/ckpt/nce/batch32nce/epoch_9/analogy.json -d sat_full
relbert-eval-analogy -m relbert_output/ckpt/nce/batch32nce/model -o relbert_output/ckpt/nce/batch32nce/model/analogy.json -d sat_full