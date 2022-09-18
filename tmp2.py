from glob import glob

for i in glob("relbert_output/models/*/epoch_*")
for p in ["a" "b" "c" "d" "e"]:
  for METHOD in "average" "average-no-mask" "mask":
      for e in range(1, 31):
          ckpt =