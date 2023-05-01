import json
import logging
import os
from tqdm import tqdm
from time import sleep
from typing import List
from statistics import mean
import pandas as pd
import openai
from datasets import load_dataset

openai.api_key = os.getenv("OPENAI_API_KEY", None)
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
all_datasets = [
    'sat_full',
    # 'u2', 'u4', 'google', 'bats',
    # 't_rex_relational_similarity', 'conceptnet_relational_similarity', 'nell_relational_similarity', 'scan'
]

def get_reply(model, text):
    while True:
        try:
            reply = openai.ChatCompletion.create(model=model, messages=[{"role": "user", "content": text}])
            break
        except Exception:
            print('Rate limit exceeded. Waiting for 10 seconds.')
            sleep(10)
    return reply['choices'][0]['message']['content']


def get_input(options: List, query: List, prompt_type: str = "1"):
    q_h, q_t = query
    if prompt_type == '1':
        prefix = "Answer the question by choosing the correct option. Which of the following is an analogy?\n"
        for n, (h, t) in enumerate(options):
            tmp = "<1> is to <2> what <3> is to <4>".replace("<1>", q_h).replace("<2>", q_t).replace("<3>", h).replace("<4>", t)
            prefix += f"{n + 1}) {tmp}\n"
    elif prompt_type == "2":
        prefix = "Answer the question by choosing the correct option. Which of the following is analogous to <1> and <2>?\n".replace("<1>", q_h).replace("<2>", q_t)
        for n, (h, t) in enumerate(options):
            tmp = "<3> and <4>".replace("<3>", h).replace("<4>", t)
            prefix += f"{n + 1}) {tmp}\n"
    elif prompt_type == "3":
        prefix = "Only one of the following statements is correct. Please answer by choosing the correct option.\n"
        for n, (h, t) in enumerate(options):
            tmp = "The relation between <1> and <2> is analogous to the relation between <3> and <4>.".replace("<1>", q_h).replace("<2>", q_t).replace("<3>", h).replace("<4>", t)
            prefix += f"{n + 1}) {tmp}\n"
    else:
        raise ValueError("unknown prompt type")
    prefix += "The answer is"
    return prefix


def get_chat(model, data_name, prompt_id):
    dataset = load_dataset("relbert/analogy_questions", data_name, split="test")
    dataset_prompt = [get_input(query=x['stem'], options=x['choice'], prompt_type=prompt_id) for x in dataset]
    output_list = []
    answer = dataset['answer']
    for n, i in tqdm(enumerate(dataset_prompt)):
        reply = get_reply(model, i)
        output_list.append({"reply": reply, "input": i, "model": model, "answer": answer[n]})
    return output_list


if __name__ == '__main__':
    os.makedirs('results/chat', exist_ok=True)

    # compute perplexity
    result = []
    for target_model in ['gpt-3.5-turbo', 'gpt-4']:
        for target_data_name in all_datasets:
            # for _prompt in ["1", "2", "3"]:
            for _prompt in ["1"]:
                scores_file = f"results/chat/{target_model}.{target_data_name}.{_prompt}.json"
                if not os.path.exists(scores_file):
                    logging.info(f"[COMPUTING PERPLEXITY] model: `{target_model}`, data: `{target_data_name}`, prompt: `{_prompt}`")
                    scores_dict = get_chat(target_model, target_data_name, prompt_id=_prompt)
                    with open(scores_file, 'w') as f:
                        json.dump(scores_dict, f)
                with open(scores_file) as f:
                    scores_dict = json.load(f)
                accuracy = []
                for s in scores_dict:
                    out = s['input'].split("\n")[1:-1]
                    if s['reply'][0] in ['1', '2', '3', '4', '5']:
                        pred = int(s['reply'][0]) - 1
                    elif s['reply'].replace("option ", "")[0] in ['1', '2', '3', '4', '5']:
                        pred = int(s['reply'].replace("option ", "")[0]) - 1
                    elif any(s['reply'][:-1] in o for o in out):
                        pred = int([o for o in out if s['reply'][:-1] in o][0][0]) - 1
                    else:
                        print(out)
                        print(s['reply'])
                        raise ValueError("unknown reply")
                    accuracy.append(int(int(s['answer']) == pred))

                result.append({"accuracy": mean(accuracy), "model": target_model, "data": target_data_name, "prompt": _prompt})
    df = pd.DataFrame(result)
    print(df)
