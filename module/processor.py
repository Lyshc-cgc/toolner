import os
import copy
import random
import math
import jsonlines
import multiprocess

from tqdm import tqdm
from datasets import load_dataset, load_from_disk, Dataset
from module import func_util as fu
from module.label import Label

logger = fu.get_logger('Processor')

class Processor(Label):
    """
    The Processor class is used to process the data.
    """
    def __init__(self, data_cfg, labels_cfg, natural_form=False):
        """
        Initialize the Processor class.
        :param data_cfg: the data processing config from the config file.
        :param labels_cfg: the configuration of the label_cfgs.
        :param natural_form: whether the labels are in natural language form.
        """
        super().__init__(labels_cfg, natural_form)
        self.config = data_cfg
        self.natural_flag = 'natural' if natural_form else 'bio'  # use natural-form or bio-form

    def _get_span_and_tags(self, tokens, tags):
        """
        Get the span and span tags of the sentence, given the tokens and token tags.
        :param tokens: tokens of the sentence
        :param tags: tags for each token
        :return:
        """
        instance_spans = []  # store spans for each instance
        instance_spans_labels = []  # store labels for each span of each instance
        idx = 0
        span = []  # store tokens in a span
        pre_tag = 0  # the previous tag
        start, end = 0, 0  # the start/end index for a span

        while idx < len(tokens):
            tag = tags[idx]
            if tag != 0:
                if pre_tag != 0 and self.covert_tag2id[tag] == self.covert_tag2id[pre_tag]:  # the token is in the same span
                    # append the token into the same span
                    span.append(tokens[idx])
                    end = idx + 1  # exclusive
                else:  # the previous is a 'O' token or previous token is not in the same span
                    # store the previous span
                    if len(span) > 0:
                        instance_spans.append((str(start), str(end), ' '.join(span)))
                        span_tag = tags[start]  # the label of the span, we use the label of the first token in the span
                        instance_spans_labels.append(self.covert_tag2id[span_tag])
                    # init a new span
                    span.clear()
                    span.append(tokens[idx])
                    start = idx
                    end = idx + 1  # exclusive
            pre_tag = tag
            idx += 1
        # store the last span
        if len(span) > 0:
            instance_spans.append((str(start), str(end), ' '.join(span)))
            instance_spans_labels.append(self.covert_tag2id[tags[start]])
        return instance_spans, instance_spans_labels

    def data_format_span(self, instances):
        """
        Get the span from gold annotated spans.
        :param instances: Dict[str, List], A batch of instances.
        :return:
        """

        # init the result
        res_tokens = []  # store the tokens of the instances
        res_tags = []  # store the tags of the instances
        res_spans_labels = []  # store the gold spans and labels of the instances

        # main process
        tokens_filed, ner_tags_field = self.config['tokens_field'], self.config['ner_tags_field']
        all_raw_tokens, all_raw_tags = instances[tokens_filed], instances[ner_tags_field]
        # 1. Some preparations

        # 1.2. covert tokens to sentence
        sents = [' '.join(raw_tokens) for raw_tokens in all_raw_tokens]

        # 1.3. get batch for different settings
        if not self.config['nested']:  # flat ner
            pbar = zip(sents, all_raw_tokens, all_raw_tags)
        else:  # nested
            start_position, end_position = instances['starts'], instances['ends']
            pbar = zip(sents, all_raw_tokens, start_position, end_position, all_raw_tags)
        for instance in pbar:
            if not self.config['nested']:  # flat NER
                sent, raw_tokens, raw_tags = instance

                # 2.1.2 get gold spans and its labels
                gold_spans, gold_spans_tags = self._get_span_and_tags(raw_tokens, raw_tags)

                # element in gold_spans is in the shape of (str(start), str(end) (excluded), span)
                # element in gold_spans_tags is tag id
                # the elements' shape of res_spans_labels is like [(start, end (excluded), gold_mention_span, gold_label_id)...]

                res_tokens.append(raw_tokens)
                res_tags.append(raw_tags)
                res_spans_labels.append([(*gs, str(gst)) for gs, gst in zip(gold_spans, gold_spans_tags)])
            else:  # nested NER
                sent, raw_tokens, starts, ends, raw_tags = instance
                # 2.1.1 (optional) get the tag directly from the raw dataset
                gold_spans = []  # store gold spans for this instance
                for start, end, label_id in zip(starts, ends, raw_tags):
                    # end position is excluded
                    gold_spans.append((str(start), str(end), ' '.join(raw_tokens[start: end]), str(label_id)))
                # the elements' shape of res_spans_labels is like [(start, end (excluded), gold_mention_span, gold_label_id)...]
                res_spans_labels.append(gold_spans)
                res_tokens.append(raw_tokens)
                res_tags.append([])

        return {
            'tokens': res_tokens,
            'tags': res_tags,
            'spans_labels': res_spans_labels,  # store the gold spans and labels of the instances, shape like (start, end (excluded), gold_mention_span, gold_label_id)
        }

    def statistics(self, dataset, include_none=False):
        """
        Get the statistics of the dataset.
        :param dataset: the dataset to be analyzed.
        :param include_none: whether to include the instances without any golden entity spans. True means to include.
        :return: the statistics of the dataset.
        """
        # get the statistics of the dataset
        # check the cached
        # 1.1 get the entity number of each label

        label_nums = {k: 0 for k in self.label2id.keys() if k != 'O'}  # store the number of entities for each label
        label_indices = {k: [] for k in self.label2id.keys() if k != 'O' }  # store the index of instances for each label

        if include_none:
            label_nums['none'], label_indices['none'] = 0, []  # store the number and index of instances without any golden entity spans

        for instance in dataset:
            if include_none and len(instance['spans_labels']) == 0:
                label_nums['none'] += 1
                label_indices['none'].append(instance['id'])
                continue

            for spans_label in instance['spans_labels']:
                # shape like (start, end, gold_mention_span, gold_label_id)
                label_id = int(spans_label[-1])
                label = self.id2label[label_id]
                label_nums[label] += 1
                label_indices[label].append(instance['id'])

        # remove dunplicate indices
        for k, v in label_indices.items():
            label_indices[k] = list(set(v))

        sum_labels = sum(label_nums.values())
        label_dist = {k: v / sum_labels for k, v in label_nums.items()}

        return {
            'label_nums': label_nums,
            'label_dist': label_dist,
            'label_indices': label_indices
        }

    def process(self):
        # 0. init config
        self.config['preprocessed_dir'] = self.config['preprocessed_dir'].format(dataset_name=self.config['dataset_name'])
        self.config['continue_dir'] = self.config['continue_dir'].format(dataset_name=self.config['dataset_name'])
        self.config['ss_cache_dir'] = self.config['ss_cache_dir'].format(dataset_name=self.config['dataset_name'])

        preprocessed_dir = os.path.join(self.config['preprocessed_dir'], f'span_{self.natural_flag}')
        process_func = self.data_format_span
        # with_rank is used to determine whether to assign a value to the rank parameter in the map function
        continue_dir = os.path.join(self.config['continue_dir'], f'span_{self.natural_flag}')  # the directory to store the continued data to be annotated

        # set 'spawn' start method in the main process to parallelize computation across several GPUs when using multi-processes in the map function
        # refer to https://huggingface.co/docs/datasets/process#map
        multiprocess.set_start_method('spawn')

        # 1. check and load the cached formatted dataset
        try:
            logger.info('Try to load the preprocessed dataset from the cache...')
            preprocessed_dataset = load_from_disk(preprocessed_dir)
        except FileNotFoundError:
            # 2. format datasets
            logger.info('No cache found, start to preprocess the dataset...')
            data_path = self.config['data_path'].format(dataset_name=self.config['dataset_name'])
            # raw dataset
            preprocessed_dataset = load_dataset(data_path, num_proc=self.config['num_proc'], trust_remote_code=True)

            tokens_filed, ner_tags_field = self.config['tokens_field'], self.config['ner_tags_field']
            if not self.config['nested']:
                # for those flat datasets, we need to filter out those instances with different length of tokens and tags
                preprocessed_dataset = preprocessed_dataset.filter(lambda x: len(x[tokens_filed]) == len(x[ner_tags_field]) )
            preprocessed_dataset = preprocessed_dataset.map(process_func,
                                                            batched=True,
                                                            batch_size=self.config['batch_size'],
                                                            num_proc=self.config['num_proc'])
            # add index column
            preprocessed_dataset = preprocessed_dataset.map(lambda example, index: {"id": index}, with_indices=True)  # add index column

            os.makedirs(self.config['preprocessed_dir'], exist_ok=True)
            preprocessed_dataset.save_to_disk(preprocessed_dir)

        # 3. shuffle, split and then save the formatted dataset
        # 3.1 check the cached result
        if self.config['continue']:
            try:
                dataset = load_from_disk(continue_dir)
                return dataset
            except FileNotFoundError:
                dataset = None

        # 3.2 get the specific split of the formatted dataset
        if self.config['split'] is not None:
            dataset = preprocessed_dataset[self.config['split']]

        # 3.3 shuffle the formatted dataset
        if self.config['shuffle']:
            dataset = dataset.shuffle()

        dataset.save_to_disk(continue_dir)

        return dataset
