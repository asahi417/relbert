relbert-train -m roberta-large -n -p -s -t d --export ./relbert_output/old_model/d --mode mask

relbert-train -m roberta-large -n -p -s --export ./ckpt/roberta_mining_top_0 --custom-template "<obj> oshibori are used in summer, and <subj> oshibori in winter."
relbert-train -m roberta-large -n -p -s --export ./ckpt/roberta_mining_top_1 --custom-template "The opposite of <obj> chaining is <subj> chaining."
relbert-train -m roberta-large -n -p -s --export ./ckpt/roberta_mining_top_2 --custom-template "<subj> — Valery Vladimirovich, <obj> — Galina Ivanovna."
relbert-train -m roberta-large -n -p -s --export ./ckpt/roberta_mining_top_3 --custom-template "Trucks enter on the <obj> end and exit on the <subj> end."
relbert-train -m roberta-large -n -p -s --export ./ckpt/roberta_mining_top_4 --custom-template "Her <subj> is from Diyarbakır and her <obj> is from Kastamonu."
relbert-train -m roberta-large -n -p -s --export ./ckpt/roberta_mining_top_5 --custom-template "His <obj> is Serpil Özyavuz, and his <subj> is Mehmet Özyavuz."
relbert-train -m roberta-large -n -p -s --export ./ckpt/roberta_mining_top_6 --custom-template "Her <subj> was Caddo, and her <obj> was Potawatomi."
relbert-train -m roberta-large -n -p -s --export ./ckpt/roberta_mining_top_7 --custom-template "Her <subj> was from Kottayam and her <obj> was from Palakkad."
relbert-train -m roberta-large -n -p -s --export ./ckpt/roberta_mining_top_8 --custom-template "And then, suddenly, it's <obj> with the old and <subj> with the new."
relbert-train -m roberta-large -n -p -s --export ./ckpt/roberta_mining_top_9 --custom-template "There are <obj> things and there are <subj> things."

relbert-train -m roberta-large -n -p -s --export ./ckpt/roberta_mining_bottom_0 --custom-template "She came <obj> as bisexual <subj> 1984."
relbert-train -m roberta-large -n -p -s --export ./ckpt/roberta_mining_bottom_1 --custom-template "He carried <obj> structural reforms <subj> Pancrase."
relbert-train -m roberta-large -n -p -s --export ./ckpt/roberta_mining_bottom_2 --custom-template "He came <obj> as bisexual <subj> 2016."
relbert-train -m roberta-large -n -p -s --export ./ckpt/roberta_mining_bottom_3 --custom-template "She dropped <obj> of movies <subj> 1937."
relbert-train -m roberta-large -n -p -s --export ./ckpt/roberta_mining_bottom_4 --custom-template "Limited excavation carried <obj> on <subj> 1996."
relbert-train -m roberta-large -n -p -s --export ./ckpt/roberta_mining_bottom_5 --custom-template "The concerts sold <obj> well <subj> advance."
relbert-train -m roberta-large -n -p -s --export ./ckpt/roberta_mining_bottom_6 --custom-template "Warren also came <obj> as lesbian <subj> 1974."
relbert-train -m roberta-large -n -p -s --export ./ckpt/roberta_mining_bottom_7 --custom-template "He came <obj> publicly <subj> 1993."
relbert-train -m roberta-large -n -p -s --export ./ckpt/roberta_mining_bottom_8 --custom-template "<obj> of Doors <subj> Florida."
relbert-train -m roberta-large -n -p -s --export ./ckpt/roberta_mining_bottom_9 --custom-template "Mary came <obj> as bisexual <subj> 2008."


relbert-eval -c 'ckpt/*/*' --export-file ./output/accuracy.analogy.csv