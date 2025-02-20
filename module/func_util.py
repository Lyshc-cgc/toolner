import os
import math
import random
import re
import yaml
import itertools
import logging
import requests
import json
import pandas as pd
import numpy as np
from omegaconf import DictConfig
from collections import Counter
from yaml import SafeLoader
from tqdm import tqdm

def set_seed(seed: int = 22):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # if using CUDA
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # make sure that cuda operations are deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_logger(name, level=logging.INFO):
    logging.basicConfig(level=level, format='[%(asctime)s | %(levelname)s | %(name)s | %(filename)s:%(lineno)d ] -- %(message)s')
    logger = logging.getLogger(name)
    return logger

def get_config(cfg_file):
    """
    Get the configuration from the configuration file.

    :param cfg_file: str, the path to the configuration file. YAML format is used.
    :return: dict, the configuration.
    """
    with open(cfg_file, 'r') as f:
        config = yaml.load(f, Loader=SafeLoader)
    return config

def batched(iterable, n):
    """
    Yield successive n-sized batches from iterable. It's a generator function in itertools module of python 3.12.
    https://docs.python.org/3.12/library/itertools.html#itertools.batched
    However, it's not available in python 3.10. So, I have to implement it here.

    :param iterable:
    :param n:
    :return:
    """
    # batched('ABCDEFG', 3) → ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while batch := tuple(itertools.islice(it, n)):
        yield batch

def compute_span_f1(gold_spans, pred_spans):
    """
    Compute the confusion matrix, span-level metrics such as precision, recall and F1-micro.
    :param gold_spans: the gold spans.
    :param pred_spans: the spans predicted by the model.
    :return:
    """
    true_positive, false_positive, false_negative = 0, 0, 0
    for span_item in pred_spans:
        if span_item in gold_spans:
            true_positive += 1
            gold_spans.remove(span_item)
        else:
            false_positive += 1

    # these entities are not predicted.
    false_negative += len(gold_spans)

    if true_positive + false_negative == 0:
        recall = 0
    else:
        recall = true_positive / (true_positive + false_negative)
    if true_positive + false_positive == 0:
        precision = 0
    else:
        precision = true_positive / (true_positive + false_positive)

    if recall + precision == 0:
        f1 = 0
    else:
        f1 = precision * recall * 2 / (recall + precision)

    return {
        'true_positive': true_positive,
        'false_positive': false_positive,
        'false_negative': false_negative,
        'recall': recall,
        'precision': precision,
        'f1': f1
    }

def compute_span_f1_by_labels(gold_spans, pred_spans, id2label, res_file):
    """
    Compute the confusion matrix, span-level metrics such as precision, recall and F1-micro for each label.
    :param gold_spans: the gold spans.
    :param pred_spans: the predicted spans.
    :param id2label: a map from label id to the label name.
    :param res_file: the file to save the results.
    :return:
    """
    # one record for one label
    # every record conclude all information we need
    # 1) Label, the label name
    # 2) Gold count, the number of gold spans for this label
    # 3) Gold rate, the proportion of gold spans to the total gold spans for this label
    # 4) Pred count, the number of predicted spans for this label
    # 5) Pred rate, the proportion of predicted spans to the total predicted spans for this label
    # 6) TP, the true positive for this label
    # 7) FP, the false positive for this label
    # 8) FN, the false negative for this label
    # 9) Pre, the precision for this label
    # 10) Rec, the recall for this label
    # 11) F1, the F1-micro for this label

    label_record = {}
    for lb in id2label.values():
        # Label 'O', representing an instance without any entities
        label_record[lb] = {"Label": lb, "Gold count": 0, "Gold rate": 0, "Pred count": 0, "Pred rate": 0,
                           "TP": 0, "FP": 0, "FN": 0, "P": 0, "R": 0, "F1": 0}

    for gold_span in gold_spans:
        if len(gold_span) == 0:
            # Label 'O', representing an instance without any entities
            label_record['O']["Gold count"] += 1
            continue
        label = gold_span[-1]  # shape like (start, end, span, label)
        label_record[label]["Gold count"] += 1

    ood_type_preds = []
    ood_mention_preds = []
    for pred_span in tqdm(pred_spans, desc="compute metric"):
        if len(pred_span) == 0:
            label = 'O'
        else:
            mention, label = pred_span[-2], pred_span[-1]  # shape like (start, end, mention span, label)
            # ood type
            if label not in id2label.values():
                ood_type_preds.append({label: mention})
                continue
        label_record[label]["Pred count"] += 1
        # ood mention,
        # if tmp_mention not in item["sentence"]:
        #     ood_mention_preds.append({tmp_mention: tmp_type})
        #     continue

        if pred_span in gold_spans:
            label_record[label]["TP"] += 1
            gold_spans.remove(pred_span)

    # the total metrics
    n_gold_tot = sum([x["Gold count"] for x in label_record.values()])
    n_pred_tot = sum([x["Pred count"] for x in label_record.values()])
    true_positive_total = sum([x["TP"] for x in label_record.values()])
    false_positive_total = n_pred_tot - true_positive_total
    false_negative_total = n_gold_tot - true_positive_total
    precision = true_positive_total / n_pred_tot if n_pred_tot else 0
    recall = true_positive_total / n_gold_tot if n_gold_tot else 0
    if precision and recall:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0
    precision_total = round(precision, 4) * 100
    recall_total = round(recall, 4) * 100
    f1_total = round(f1, 4) * 100

    # metrics for each label
    for l in label_record:
        gold_count = label_record[l]["Gold count"]
        pred_count = label_record[l]["Pred count"]
        true_positive = label_record[l]["TP"]
        false_positive = pred_count - true_positive
        false_negative = gold_count - true_positive

        gold_rate = gold_count / n_gold_tot if n_gold_tot else 0
        pred_rate = pred_count / n_pred_tot if n_pred_tot else 0
        gold_rate = round(gold_rate, 4) * 100
        pred_rate = round(pred_rate, 4) * 100

        pre = true_positive / pred_count if pred_count else 0
        rec = true_positive / gold_count if gold_count else 0
        if pre and rec:
            f1 = 2 * pre * rec / (pre + rec)
        else:
            f1 = 0
        pre = round(pre, 4) * 100
        rec = round(rec, 4) * 100
        f1 = round(f1, 4) * 100

        label_record[l]["Gold rate"] = gold_rate
        label_record[l]["Pred rate"] = pred_rate
        label_record[l]["TP"] = true_positive
        label_record[l]["FP"] = false_positive
        label_record[l]["FN"] = false_negative
        label_record[l]["P"] = pre
        label_record[l]["R"] = rec
        label_record[l]["F1"] = f1

    label_record["Total"] = {"Label": "ToTal", "Gold count": n_gold_tot, "Gold rate": 100, "Pred count": n_pred_tot,
                            "Pred rate": 100, "TP": true_positive_total, "FP": false_positive_total, "FN": false_negative_total,
                             "P": precision_total, "R": recall_total, "F1": f1_total}

    # convert to dataframe
    df_metrics = pd.DataFrame(list(label_record.values()))
    print(f"===== Metrics for each label =====\n{df_metrics}")
    # cache the results
    df_metrics.to_csv(res_file, index=False)

def find_span(text: str, span: str):
    """
    Find the span in the text.
    :param text: str, the text.
    :param span: str, the mention.
    :return: list, the list of spans.
    """
    if not span:
        return []
    res_spans = []
    # Find the start character index and end character index of the first matched span.
    re_span = re.escape(str(span))  # escape special characters in the span
    pattern_0 = r"\b(" + re_span + r")\b"  # match the whole span after escaping special characters
    pattern_1 = r"\s(" + re_span + r")\s"  # match the span surrounded by spaces after escaping special characters
    patterns = [pattern_0, pattern_1]
    res_matches = []
    for pattern in patterns:
        matches = re.finditer(pattern, text)
        res_matches += [match for match in matches]

    for match in res_matches:
        start_ch_idx, end_ch_idx = match.span(1)  # get the capture group 1
        # To get the start position of the first word of the matched NP span,
        # we just need to count the number of spaces before the start character
        start = text[:start_ch_idx].count(' ')

        # To get the end position of the last word of the matched NP span,
        # we just need to count the number of spaces before the end character
        end = text[:end_ch_idx].count(' ') + 1  # end position of the NP span, excluded
        res_spans.append((start, end, span))

    return res_spans

def get_label_subsets(labels, sub_size, repeat_num=1, fixed_subsets=None):
    """
    Get the subsets of the labels.
    :param labels: list, the list of labels.
    :param sub_size: int or float (<1), the size of the label subset.
    :param repeat_num: the number of times to repeat each label.
    :param fixed_subsets: a list of lists or tuples, the fixed subsets. we randomly sample the rest of the labels. e.g., [['PER', 'ORG'], ['LOC', 'GPE'],...]
    :return: list, the list of subsets.
    """
    if 0 < sub_size < 1:
        sub_size = math.floor(len(labels) * sub_size)
        if sub_size < 1:
            sub_size = 1

    label_subsets = []
    for _ in range(repeat_num):
        random.shuffle(labels)
        if fixed_subsets:
            labels = [l for l in labels if l not in fixed_subsets]  # filter out labels in the fixed subsets
            label_subsets += fixed_subsets
        label_subsets += list(batched(labels, sub_size))  # batch method return a generator
    return label_subsets

def get_label_mention_pairs(original_pairs, label_mention_map_portion, id2label):
    """
    Get the label-mention pairs, where the label-mention pairs are partially correct.
    :param original_pairs: the original label-mention pairs.
    :param label_mention_map_portion: the portion of the corrected label-mention pair. Default is 1, which means all the label-mention pairs are correct.
    :param id2label: a map from label id to the label name.
    :return:
    """
    random.shuffle(original_pairs)
    wrong_pairs_num = int(len(original_pairs) * (1-label_mention_map_portion))
    tmp_wrong_pairs, correct_pairs = original_pairs[:wrong_pairs_num], original_pairs[wrong_pairs_num:]
    wrong_pairs = []
    for start, end, entity_mention, label_id in tmp_wrong_pairs:
        label_ids = list(id2label.keys())
        label_ids.remove(int(label_id))
        wrong_label_id = random.choice(label_ids)
        wrong_pairs.append((start, end, entity_mention, wrong_label_id))
    res_pairs = correct_pairs + wrong_pairs
    return res_pairs

def remove_duplicated_label_sets(label_sets: list):
    """
    Remove the duplicated label sets.
    :param label_sets: the label sets
    :return:
    """
    duplicated_id = []  # store the idx of the duplicated label sets
    for i in range(len(label_sets)):
        if i in duplicated_id:  # skip the idx that has been in the duplicated_id
            continue
        for j in range(i + 1, len(label_sets)):
            if j in duplicated_id:  # skip the idx that has been in the duplicated_id
                continue
            if Counter(label_sets[i]) == Counter(label_sets[j]):
                duplicated_id.append(j)
    label_sets = [label_sets[i] for i in range(len(label_sets)) if i not in duplicated_id]
    return label_sets

def request_dify_chat(base_url,
                      token,
                      query,
                      inputs={},
                      response_mode="streaming",
                      app_mode='other'
                      ):
    """
    Request to Dify Chat API

    :param base_url: dify api base url
    :param token: token for the request
    :param query: user query
    :param inputs: a dict, input key-value pairs
    :param response_mode: response mode, 'streaming' or 'blocking'
    :param app_mode: application mode, 'agent', 'other'
    :return:
    """
    assert response_mode in {'streaming', 'blocking'}, f"response_mode should be 'streaming' or 'blocking', but got {response_mode}"
    if app_mode == 'agent':
        response_mode = 'streaming'
    assert app_mode in {'agent', 'other', 'chat'}, f"app_mode should be 'agent', 'other' or 'chat', but got {app_mode}"
    url = f'{base_url}/chat-messages'
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
    data = {
        "inputs": inputs,
        "query": query,
        "response_mode": response_mode,
        "conversation_id": "",
        "user": "abc-123",
        # "files": [
        #     {
        #         "type": "image",
        #         "transfer_method": "remote_url",
        #         "url": "https://cloud.dify.ai/logo/logo-site.png"
        #     }
        # ]
    }
    stream_flag = True if response_mode == "streaming" else False
    response = requests.post(url, headers=headers, json=data, stream=stream_flag)
    chunks = response.text.split('\n\n')
    resp_message = ''
    event_type = 'agent_message' if app_mode == 'agent' else 'message'
    for chunk in chunks:
        try:
            chunk_json = json.loads(chunk[5:])  # remove the first 5 characters
        except Exception as e:
            continue
        if chunk_json['event'] == event_type:
            resp_message += chunk_json['answer'] + '\n'
    return resp_message

def request_dify_completion(base_url,
                            inputs,
                            token,
                            response_mode="streaming"
                            ):
    """
    Request to Dify Completion API

    :param base_url: dify api base url
    :param inputs: a dict, input key-value pairs
    :param token: token for the request
    :param response_mode: response mode, 'streaming' or 'blocking'
    :return:
    """
    assert response_mode in {'streaming', 'blocking'}, f"response_mode should be 'streaming' or 'blocking', but got {response_mode}"
    url = f'{base_url}/completion-messages'
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
    assert 'query' in inputs, "inputs should contain 'query' key"
    assert inputs['query'] != '', "query should not be empty!"

    data = {
        "inputs": inputs,
        "response_mode": response_mode,
        "user": "abc-123",
    }
    stream_flag = True if response_mode == "streaming" else False
    response = requests.post(url, headers=headers, json=data, stream=stream_flag)
    chunks = response.text.split('\n\n')
    resp_message = ''
    for chunk in chunks:
        try:
            chunk_json = json.loads(chunk[5:])  # remove the first 5 characters
        except Exception as e:
            continue
        if chunk_json['event'] == 'message':
            resp_message += chunk_json['answer'] + '\n'
    return resp_message

def clean_format(message):
    message = re.sub(r'\n', '', message)
    message = re.sub(r"\s+", " ", message)
    return message

def init_file_path(config: DictConfig, file_dir, file_postfix_name):
    augment_method_postfix = ''
    max_entities_postfix = ''
    generate_method_postfix = ''
    if 'augment_method' in config and config.augment_method is not None:
        augment_method_postfix = f'-{config.augment_method}'
    if 'max_entities' in config and config.max_entities is not None:
        max_entities_postfix = f'-e{config.max_entities}'
    if 'generate_method' in config and config.generate_method == 'fixed':
        generate_method_postfix = config.generate_method
    if ((augment_method_postfix != '' or max_entities_postfix !='' or generate_method_postfix != '')
            and not file_postfix_name.startswith('_')):
        file_postfix_name = f'_{file_postfix_name}'
    file_path = os.path.join(
        file_dir,
        f'{generate_method_postfix}{max_entities_postfix}{augment_method_postfix}{file_postfix_name}'
    )
    return file_path
