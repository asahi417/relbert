# check reproducibility
relbert-train -o "tmp/test_1" -m "roberta-base" -b 32 -e 1 --loss nce -r 0.000005
relbert-eval-analogy -m "tmp/test_1/model" -o "tmp/test_1/model" -d sat_full

relbert-train -o "tmp/test_2" -m "roberta-base" -b 32 -e 1 --loss nce -r 0.000005
relbert-eval-analogy -m "tmp/test_2/model" -o "tmp/test_2/model.tm" -d sat_full