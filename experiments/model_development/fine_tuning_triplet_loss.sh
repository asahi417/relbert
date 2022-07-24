# Fine-tuning RelBERT on triplet loss with the old version of RelBERT library
git clone https://github.com/asahi417/relbert
cd relbert
git checkout emnlp_2021
pip install -e .

# train
relbert-train -m roberta-large -n -p -s -t a --export ./relbert_output/ckpt_triplet/roberta_custom_a --mode average
relbert-train -m roberta-large -n -p -s -t b --export ./relbert_output/ckpt_triplet/roberta_custom_b --mode average
relbert-train -m roberta-large -n -p -s -t c --export ./relbert_output/ckpt_triplet/roberta_custom_c --mode average
relbert-train -m roberta-large -n -p -s -t d --export ./relbert_output/ckpt_triplet/roberta_custom_d --mode average
relbert-train -m roberta-large -n -p -s -t e --export ./relbert_output/ckpt_triplet/roberta_custom_e --mode average
# eval
relbert-eval -c './relbert_output/ckpt_triplet/*/epoch_1' --export-file ./accuracy.triplet.analogy.csv
relbert-eval -c './relbert_output/ckpt_triplet/*/epoch_1' --type classification --export-file ./accuracy.triplet.classification.csv
# push
relbert-push-to-hub -o relbert -m "roberta_custom_a/epoch_1" -a "relbert-roberta-large-semeval2012-average-prompt-a-triplet"
relbert-push-to-hub -o relbert -m "roberta_custom_b/epoch_1" -a "relbert-roberta-large-semeval2012-average-prompt-b-triplet"
relbert-push-to-hub -o relbert -m "roberta_custom_c/epoch_1" -a "relbert-roberta-large-semeval2012-average-prompt-c-triplet"
relbert-push-to-hub -o relbert -m "roberta_custom_d/epoch_1" -a "relbert-roberta-large-semeval2012-average-prompt-d-triplet"
relbert-push-to-hub -o relbert -m "roberta_custom_e/epoch_1" -a "relbert-roberta-large-semeval2012-average-prompt-e-triplet"

# train
relbert-train -m roberta-large -n -p -s -t a --export ./relbert_output/ckpt_triplet_mask/roberta_custom_a --mode mask
relbert-train -m roberta-large -n -p -s -t b --export ./relbert_output/ckpt_triplet_mask/roberta_custom_b --mode mask
relbert-train -m roberta-large -n -p -s -t c --export ./relbert_output/ckpt_triplet_mask/roberta_custom_c --mode mask
relbert-train -m roberta-large -n -p -s -t d --export ./relbert_output/ckpt_triplet_mask/roberta_custom_d --mode mask
relbert-train -m roberta-large -n -p -s -t e --export ./relbert_output/ckpt_triplet_mask/roberta_custom_e --mode mask
# eval
relbert-eval -c './relbert_output/ckpt_triplet_mask/*/epoch_1' --export-file ./accuracy.triplet.mask.analogy.csv
relbert-eval -c './relbert_output/ckpt_triplet_mask/*/epoch_1' --type classification --export-file ./accuracy.triplet.mask.classification.csv
# push
relbert-push-to-hub -o relbert -m "roberta_custom_a/epoch_1" -a "relbert-roberta-large-semeval2012-mask-prompt-a-triplet"
relbert-push-to-hub -o relbert -m "roberta_custom_b/epoch_1" -a "relbert-roberta-large-semeval2012-mask-prompt-b-triplet"
relbert-push-to-hub -o relbert -m "roberta_custom_c/epoch_1" -a "relbert-roberta-large-semeval2012-mask-prompt-c-triplet"
relbert-push-to-hub -o relbert -m "roberta_custom_d/epoch_1" -a "relbert-roberta-large-semeval2012-mask-prompt-d-triplet"
relbert-push-to-hub -o relbert -m "roberta_custom_e/epoch_1" -a "relbert-roberta-large-semeval2012-mask-prompt-e-triplet"

# train
relbert-train -m roberta-large -n -p -s -t a --export ./relbert_output/ckpt_triplet_mask/roberta_custom_a --mode average_no_mask
relbert-train -m roberta-large -n -p -s -t b --export ./relbert_output/ckpt_triplet_mask/roberta_custom_b --mode average_no_mask
relbert-train -m roberta-large -n -p -s -t c --export ./relbert_output/ckpt_triplet_mask/roberta_custom_c --mode average_no_mask
relbert-train -m roberta-large -n -p -s -t d --export ./relbert_output/ckpt_triplet_mask/roberta_custom_d --mode average_no_mask
relbert-train -m roberta-large -n -p -s -t e --export ./relbert_output/ckpt_triplet_mask/roberta_custom_e --mode average_no_mask
# eval
relbert-eval -c './relbert_output/ckpt_triplet_no_mask/*/epoch_1' --export-file ./accuracy.triplet.no_mask.analogy.csv
relbert-eval -c './relbert_output/ckpt_triplet_no_mask/*/epoch_1' --type classification --export-file ./accuracy.triplet.no_mask.classification.csv
# push
relbert-push-to-hub -o relbert -m "roberta_custom_a/epoch_1" -a "relbert-roberta-large-semeval2012-average-no-mask-prompt-a-triplet"
relbert-push-to-hub -o relbert -m "roberta_custom_b/epoch_1" -a "relbert-roberta-large-semeval2012-average-no-mask-prompt-b-triplet"
relbert-push-to-hub -o relbert -m "roberta_custom_c/epoch_1" -a "relbert-roberta-large-semeval2012-average-no-mask-prompt-c-triplet"
relbert-push-to-hub -o relbert -m "roberta_custom_d/epoch_1" -a "relbert-roberta-large-semeval2012-average-no-mask-prompt-d-triplet"
relbert-push-to-hub -o relbert -m "roberta_custom_e/epoch_1" -a "relbert-roberta-large-semeval2012-average-no-mask-prompt-e-triplet"



