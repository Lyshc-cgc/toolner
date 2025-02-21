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
    def __init__(self, config):
        """
        Initialize the Processor class.
        :param config: the configuration of the project
        """
        super().__init__(config.dataset, config.natural_form)
        self.data_config = config.dataset
        self.natural_flag = 'natural' if config.natural_form else 'bio'  # use natural-form or bio-form

    def _get_span_and_labels(self, tokens, tags):
        """
        Get the spans and their labels of the sentence, given the tokens and token tags.
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
                        # self.covert_tag2id[span_tag], covert the tag to the new label id
                        instance_spans_labels.append(self.id2label[self.covert_tag2id[span_tag]])
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
            instance_spans_labels.append(self.id2label[self.covert_tag2id[tags[start]]])
        return instance_spans, instance_spans_labels

    def data_format_span(self, instances):
        """
        Get the span from gold annotated spans.
        :param instances: Dict[str, List], A batch of instances.
        :return:
        """

        # init the result
        res_sentences = []  # store the sentences of the instances
        res_tags = []  # store the tags of the instances
        res_spans_labels = []  # store the gold spans and labels of the instances

        # main process
        tokens_filed, ner_tags_field = self.data_config['tokens_field'], self.data_config['ner_tags_field']
        all_raw_tokens, all_raw_tags = instances[tokens_filed], instances[ner_tags_field]
        # 1. Some preparations

        # 1.2. covert tokens to sentence
        sents = [' '.join(raw_tokens) for raw_tokens in all_raw_tokens]

        # 1.3. get batch for different settings
        if not self.data_config['nested']:  # flat ner
            pbar = zip(sents, all_raw_tokens, all_raw_tags)
        else:  # nested
            start_position, end_position = instances['starts'], instances['ends']
            pbar = zip(sents, all_raw_tokens, start_position, end_position, all_raw_tags)
        for instance in pbar:
            if not self.data_config['nested']:  # flat NER
                sent, raw_tokens, raw_tags = instance

                # 2.1.2 get gold spans and its labels
                gold_spans, gold_spans_labels = self._get_span_and_labels(raw_tokens, raw_tags)

                # element in gold_spans is in the shape of (str(start), str(end) (excluded), span)
                # element in gold_spans_tags is tag id
                # the elements' shape of res_spans_labels is like [(start, end (excluded), gold_mention_span, gold_label)...]
                if len(gold_spans_labels) == 0:  # empty instance
                    # Randomly decide whether to include this instance based on the desired ratio
                    if random.random() < self.data_config.empty_ration:
                        res_sentences.append(sent)
                        res_tags.append(raw_tags)
                        res_spans_labels.append([(*gs, str(gst)) for gs, gst in zip(gold_spans, gold_spans_labels)])
                else:
                    res_sentences.append(sent)
                    res_tags.append(raw_tags)
                    res_spans_labels.append([(*gs, str(gst)) for gs, gst in zip(gold_spans, gold_spans_labels)])
            else:  # nested NER
                sent, raw_tokens, starts, ends, raw_tags = instance
                # 2.1.1 (optional) get the tag directly from the raw dataset
                gold_spans = []  # store gold spans for this instance
                for start, end, label_id in zip(starts, ends, raw_tags):
                    # end position is excluded
                    gold_spans.append((str(start), str(end), ' '.join(raw_tokens[start: end]), str(label_id)))
                if len(gold_spans) == 0:  # empty instance
                    # Randomly decide whether to include this instance based on the desired ratio
                    if random.random() < self.data_config.empty_ration:
                        res_sentences.append(sent)
                        res_tags.append([])
                        res_spans_labels.append(gold_spans)

                # the elements' shape of res_spans_labels is like [(start, end (excluded), gold_mention_span, gold_label)...]
                res_sentences.append(sent)
                res_tags.append([])
                res_spans_labels.append(gold_spans)

        return {
            'sentences': res_sentences,
            'tags': res_tags,
            'spans_labels': res_spans_labels,  # store the gold spans and labels of the instances, shape like (start, end (excluded), gold_mention_span, gold_label)
        }

    def process(self):
        # 0. init config
        preprocessed_dir = os.path.join(self.data_config['preprocessed_dir'], f'span_{self.natural_flag}')
        process_func = self.data_format_span

        # 1. check and load the cached formatted dataset
        try:
            logger.info('Try to load the preprocessed dataset from the cache...')
            preprocessed_dataset = load_from_disk(preprocessed_dir)
        except FileNotFoundError:
            # 2. format datasets
            logger.info('No cache found, start to preprocess the dataset...')

            # raw dataset
            preprocessed_dataset = load_dataset(
                self.data_config.file_path,
                num_proc=self.data_config.num_proc,
                trust_remote_code=True)

            tokens_filed, ner_tags_field = self.data_config.tokens_field, self.data_config.ner_tags_field
            if not self.data_config.nested:
                # for those flat datasets, we need to filter out those instances with different length of tokens and tags
                preprocessed_dataset = preprocessed_dataset.filter(lambda x: len(x[tokens_filed]) == len(x[ner_tags_field]) )
            preprocessed_dataset = preprocessed_dataset.map(
                process_func,
                batched=True,
                batch_size=self.data_config.data_batch_size,
                num_proc=self.data_config.num_proc,
                remove_columns=preprocessed_dataset['train'].column_names
            )
            # add index column
            preprocessed_dataset = preprocessed_dataset.map(lambda example, index: {"id": index}, with_indices=True)  # add index column

            os.makedirs(self.data_config.preprocessed_dir, exist_ok=True)
            preprocessed_dataset.save_to_disk(preprocessed_dir)

        # 3. shuffle, split and then save the formatted dataset
        # 3.1 check the cached result
        if 'continue_dir' in self.data_config:
            # use the dataset last time
            continue_dir = os.path.join(self.data_config.continue_dir,
                                        f'span_{self.natural_flag}')  # the directory to store the continued data to be annotated
            try:
                dataset = load_from_disk(continue_dir)
                return dataset
            except FileNotFoundError:
                dataset = None

        # 3.2 get the specific split of the formatted dataset
        if self.data_config.split is not None:
            dataset = preprocessed_dataset[self.data_config.split]

        dataset.save_to_disk(continue_dir)

        return dataset
