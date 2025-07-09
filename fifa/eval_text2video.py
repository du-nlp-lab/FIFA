import json
import os
import pandas as pd
import string
from typing import Any, Callable, Dict, List, Optional
from pathlib import Path
from utils_text2video import generate_dsg
import tqdm
import time
from parse_utils import parse_tuple_output, parse_dependency_output, parse_question_output
import re


# import video_qa.video_question_answering_internvl as video_question_answering


def eval_text2video(data, save=False, n_parallel_workers=1, cache_dir="./results", llm=None, vqamodel=None):
    id2prompts = {}

    for i, item in enumerate(data):
        id2prompts[i] = {}
        id2prompts[i]['input'] = item['prompt']
        id2prompts[i]['video'] = item['video_path']

    id2tuple_outputs, id2question_outputs, id2dependency_outputs = generate_dsg(
        id2prompts,
        llm.completion,
        verbose=False,
        N_parallel_workers=n_parallel_workers,
    )

    if not os.path.exists(cache_dir) and save:
        os.makedirs(cache_dir)
    if save:
        with open(f"{cache_dir}/id2tuple_outputs_.json", "w") as f:
            json.dump(id2tuple_outputs, f, indent="\t")
        with open(f"{cache_dir}/id2question_outputs_.json", "w") as f:
            json.dump(id2question_outputs, f, indent="\t")
        with open(f"{cache_dir}/id2dependency_outputs_.json", "w") as f:
            json.dump(id2dependency_outputs, f, indent="\t")

    results_vqa = {}
    for key in tqdm.tqdm(id2prompts.keys()):
        qid2question = parse_question_output(id2question_outputs[key]['output'])
        qid2dependency = parse_dependency_output(id2dependency_outputs[key]['output'])
        qid2tuple = parse_tuple_output(id2tuple_outputs[key]['output'])
        video_path = id2prompts[key]["video"]

        qid2answer = {}
        qid2scores = {}
        qid2validity = {}

        for id, question in qid2question.items():
            answer = vqamodel.answer(question, video_path)
            qid2answer[id] = answer
            qid2scores[id] = float('yes' in answer.lower())

        average_score_without_dep = sum(qid2scores.values()) / len(qid2scores)

        for id, parent_ids in qid2dependency.items():
            if id > len(qid2question):
                print(key)
                continue
            # zero-out scores if parent questions are answered 'no'
            any_parent_answered_no = False
            for parent_id in parent_ids:
                if parent_id == 0:
                    continue
                if qid2scores[parent_id] == 0:
                    any_parent_answered_no = True
                    break
            if any_parent_answered_no:
                qid2scores[id] = 0
                qid2validity[id] = False
            else:
                qid2validity[id] = True

        results_vqa[key] = {"qid2answer": qid2answer, "qid2scores": qid2scores, "qid2validity": qid2validity,
                            "average_score_without_dep": average_score_without_dep,
                            "average_score": sum(qid2scores.values()) / len(qid2scores)}
    print(f"final results {sum([results_vqa[key]['average_score'] for key in results_vqa]) / len(results_vqa)}")
    if save:
        with open(f"{cache_dir}/vqa_results_prompt.json", "w") as f:
            json.dump(results_vqa, f, indent="\t")




