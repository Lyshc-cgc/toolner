import ast
import re
import os
import json
import copy

from module import func_util as fu
from module.label import Label

logger = fu.get_logger('Annotation')

class Annotation(Label):
    def __init__(self, config):
        super().__init__(config.dataset, config.natural_form)
        self.config = config

    @staticmethod
    def _process_output(output_text, usr_query):
        """
        process the output text to get the label of the entity mention.
        :param output_text: the text output by the annotator model
        :param usr_query: the user's query sentence
        :return: if the single_type_prompt is True, return the predicted spans and their labels.
        """
        if not output_text:
            output_text = ''
        output_text = output_text.strip()
        out_spans = []
        pattern = r'\[(.*?)\]'  # the pattern to extract a list string
        result = re.search(pattern, output_text, re.DOTALL)  # only find the first list string
        try:
            tmp_spans = ast.literal_eval(result.group(0).strip())  # tmp_spans shapes like [(type 0, mention0),...]
            # filter the invalid spans
            tmp_spans = filter(lambda e: isinstance(e, tuple), tmp_spans)
            tmp_spans = filter(lambda e: e and len(e) == 2, tmp_spans)
        except (TypeError, Exception):  # the output_text is not valid
            pattern = r'\((.*?)\)'  # the pattern to extract a tuple
            result = re.findall(pattern, output_text, re.DOTALL)  # find all the tuples
            try:  # try to extract tuple directly
                tmp_spans = []
                for e in result:
                    e = e.split(',')
                    if len(e) == 2:
                        tmp_spans.append((e[0].strip(), e[1].strip()))
                # filter the invalid spans
                tmp_spans = filter(lambda e: isinstance(e, tuple), tmp_spans)
                tmp_spans = filter(lambda e: e and len(e) == 2, tmp_spans)
            except (TypeError, Exception):
                tmp_spans = []

        for label, mention in tmp_spans:
            founded_spans = fu.find_span(usr_query, str(mention))
            out_spans += [(str(start), str(end), span, label) for start, end, span in set(founded_spans)]
        return out_spans

    def annotate_few_shot(self, query, **kwargs):
        """
        annotate the user's query using the dify api, few-shot setting
        :param query: the user's query
        :param kwargs for few-shot setting:
            1) types_description: the type description input to system prompt
            2) demonstrations: the demonstrations input to system prompt
        :return:
        """
        assert 'types_description' in kwargs, "types_description should be provided in few-shot setting"
        assert 'demonstrations' in kwargs, "demonstrations should be provided in few-shot setting"
        assert kwargs['types_description'] is not None, "types_description should not be empty in few-shot setting"
        assert kwargs['demonstrations'] is not None, "demonstrations should not be empty in few-shot setting"
        inputs = {
            "types_description": kwargs['types_description'],
            "demonstrations": kwargs['demonstrations']
        }
        response = fu.request_dify_chat(
            base_url=self.config.dify_api.base_url,
            token=self.config.dify_api[self.config.ner_app.name],
            query=query,
            inputs=inputs,
            response_mode="streaming",
            app_mode=self.config.ner_app.app_mode
        )
        return response

    def annotate_zero_shot(self, query, **kwargs):
        """
        annotate the user's query using the dify api, zero-shot setting
        :param query: the user's query
        :param kwargs for zero-shot setting:
            1) types_description: the type description input to system prompt
        :return:
        """
        assert 'types_description' in kwargs, "types_description should be provided in zero-shot setting"
        assert kwargs['types_description'] is not None, "types_description should not be empty in zero-shot setting"
        inputs = {
            "types_description": kwargs['types_description']
        }
        response = fu.request_dify_chat(
            base_url=self.config.dify_api.base_url,
            token=self.config.dify_api[self.config.ner_app.name],
            query=query,
            inputs=inputs,
            response_mode="streaming",
            app_mode=self.config.ner_app.app_mode
        )
        return response

    def annotate(self, dataset, types_description=None, demonstrations=None, setting='few-shot'):
        """
        annotate the user's query using the dify api
        :param dataset: dataset to be annotated
        :param types_description: the type description input to system prompt
        :param demonstrations: the demonstrations input to system prompt
        :param setting: the setting of the annotation, 'few-shot' or 'zero-shot'
        :return:
        """
        # 1. init cache file
        anno_res_cache_file = fu.init_file_path(
            config=self.config,
            file_dir=self.config.cache_dir,
            file_postfix_name='anno_res_cache.json'
        )
        # 2. annotate
        pred_spans, gold_spans = [], []
        if os.path.exists(anno_res_cache_file):
            logger.info(f'load annotation results from {anno_res_cache_file}')
            with open(anno_res_cache_file, 'r') as f:
                cache_res = json.load(f)
                pred_spans = cache_res['pred_spans']
                gold_spans = cache_res['gold_spans']

        else:
            annotate_methods = {
                'few-shot': self.annotate_few_shot,
                'zero-shot': self.annotate_zero_shot
            }
            assert setting in annotate_methods, f"Invalid setting: {setting}, should be one of {annotate_methods.keys()}"

            for sentence in dataset['sentences']:
                response = annotate_methods[setting](
                    query=sentence,
                    types_description=types_description,
                    demonstrations=demonstrations
                )
                output_text = fu.clean_format(response)
                logger.info(f'input: {sentence}')
                logger.info(f'output: {output_text}')
                out_spans = self._process_output(output_text, sentence)
                logger.info(f'out_spans: {out_spans}')
                # if len(out_spans) == 0:
                #     continue
                pred_spans_in_a_query = []
                for start, end, span, label in set(out_spans):
                    if label not in self.label2id.keys():
                        continue  # skip the invalid label
                    pred_spans_in_a_query.append((str(start), str(end), span, label))
                pred_spans.append(pred_spans_in_a_query)

            # 3. cache the annotation results
            gold_spans = []
            # covert list into a tuple
            for gold_spans_in_a_query in dataset['spans_labels']:
                tmp_spans_in_a_query = []
                for span_label in gold_spans_in_a_query:
                    start, end, span, label = span_label  # span_label is a list
                    tmp_spans_in_a_query.append((str(start), str(end), span, label))
                gold_spans.append(tmp_spans_in_a_query)
            cache_res = {
                'pred_spans': pred_spans,
                'gold_spans': gold_spans
            }
            with open(anno_res_cache_file, 'w') as f:
                json.dump(cache_res, f)  # after json.dump, all tuples will be converted into lists

        # 4. evaluate the annotation results
        # flatten the pred_spans
        y_preds = []
        for pred_spans_in_a_query in pred_spans:
            if len(pred_spans_in_a_query) == 0:
                y_preds.append([])
                continue
            for e in pred_spans_in_a_query:
                y_preds.append(e)
        # flattern the y_true
        y_trues = []
        for span_label_in_a_instance in gold_spans:
            if len(span_label_in_a_instance) == 0:
                y_trues.append([])
                continue
            for e in span_label_in_a_instance:
                y_trues.append(e)
        self.evaluate(y_trues, y_preds)

    def evaluate(self, y_trues, y_preds):
        # 1. init cache file
        res_file = fu.init_file_path(
            config=self.config,
            file_dir=self.config.eval_dir,
            file_postfix_name=f'res.json'
        )
        res_by_class_file = fu.init_file_path(
            config=self.config,
            file_dir=self.config.eval_dir,
            file_postfix_name=f'res_by_class.csv'
        )
        logger.info(f'saved the evaluation results to {res_file}')
        logger.info(f'saved the evaluation results by class to {res_by_class_file}')

        # compute span-level metrics
        eval_results = fu.compute_span_f1(copy.deepcopy(y_trues), copy.deepcopy(y_preds))
        with open(res_file, 'w') as f:
            json.dump(eval_results, f, indent=4)
        fu.compute_span_f1_by_labels(
            copy.deepcopy(y_trues),
            copy.deepcopy(y_preds),
            id2label=self.id2label,
            res_file=res_by_class_file
        )
