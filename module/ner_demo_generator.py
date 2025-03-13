import ast
import re
import os
import json
import random
import math
import itertools
from tqdm import tqdm
from collections import Counter
from module import func_util as fu
from module.label import Label

logger = fu.get_logger('NERDemoGenerator')

class NERDemoGenerator(Label):
    def __init__(self, config):
        super().__init__(config.dataset, config.natural_form)
        self.config = config
        entity_types = list(self.label2id.keys())
        if 'O' in entity_types:
            entity_types.remove('O')
        self.entity_types = entity_types
        # initialize entity list
        self.entity_list = {entity_type: [] for entity_type in entity_types}

        # init for lsp augmentation
        if 'augment_method' in self.config and self.config.augment_method == 'lsp':
            assert self.config.generate_method == 'combined', "LSP augmentation only supports 'combined' generate method"
            assert 'subset_size' in self.config, "'subset_size' should be in the config"
            assert 'partition_time' in self.config, "'partition_time' should be in the config"

            self.all_labels = list(self.label2id.keys())
            if 'O' in self.all_labels:
                self.all_labels.remove('O')

            if 0 < self.config.subset_size < 1:
                subset_size = math.floor(len(self.all_labels) * self.config.subset_size)
                if subset_size < 1:
                    subset_size = 1
            else:
                subset_size = self.config.subset_size
            self.subset_size = subset_size

    def generate_initial_entities(self):
        init_entities_file = os.path.join(self.config.cache_dir, 'initial_entities.json')
        if os.path.exists(init_entities_file):
            try:
                with open(init_entities_file, "r", encoding="utf-8") as f:
                    self.entity_list = json.load(f)
                logger.info(f"Cached initial entities found! Load initial entities from {init_entities_file}")
                return
            except Exception as e:
                logger.error(f"Error during loading initial entities: {e}")

        # no cached file. generate initial entities from scratch
        for entity_type in self.entity_types:
            initial_entities = self._generate_entities(entity_type, self.config.k_shot)
            self.entity_list[entity_type].extend(initial_entities)
        logger.info("Initial Entity List:")
        logger.info(self.entity_list)

        # save initial entities to file
        try:
            with open(init_entities_file, "w", encoding="utf-8") as f:
                json.dump(self.entity_list, f, ensure_ascii=False, indent=4)
            logger.info(f"Save initial entities to {init_entities_file}")
        except Exception as e:
            logger.info(f"Error during saving initial entities: {e}")

    def _get_type_description(self, entity_type):
        """
        get the description of the entity type
        :param entity_type:
        :return:
        """
        type_description = ''
        if self.config.des_format == '' or self.config.des_format == 'empty':
            return type_description

        type_description = self.label_description[entity_type]  # full description
        if self.config.des_format == 'simple':
            type_description = type_description.split('.')[:2]
            type_description = ' '.join(type_description)

        return type_description

    def _generate_entities(self, entity_type, num_entities, retry_num=0):
        def _extract_entity(sentence):
            pattern = r"\[.*?\]"
            matches = re.findall(pattern, sentence)
            if len(matches) == 0:
                return None
            try:
                entities = ast.literal_eval(matches[0])  # check if the string is a valid python expression
                random.shuffle(entities)  # random shuffle to avoid the same or common entities
                return entities[:num_entities]
            except Exception as e:
                logger.error(f"Error during extracting entities: {e}")
                return None

        type_description = self._get_type_description(entity_type)
        inputs = {'type_description': type_description}

        generated_entities, generated_num = set(), 0
        while len(generated_entities) == 0 or len(generated_entities) < num_entities:
            if retry_num < 0:
                break
            # There is still query_entity_num entities that has not been generated
            query = f'generate {num_entities + 3} "{entity_type}" named entities or phrases. '  # 3 more to avoid the same or common entities
            res_message= fu.request_dify_chat(
                base_url=self.config.dify_api.base_url,
                token=self.config.dify_api[self.config.entity_app.name],
                query=query,
                inputs=inputs,
                response_mode="streaming",
                app_mode=self.config.entity_app.app_mode
            )
            res_message = fu.clean_format(res_message)
            extracted_entities = _extract_entity(res_message)
            retry_num -= 1
            if extracted_entities is None:
                logger.info(f"Error during extracting entities from: {res_message}")
                continue
            try:
                generated_entities.update(extracted_entities)
            except Exception as e:
                logger.error(f"Error during updating generated entities: {e}")
                continue
        generated_entities = list(generated_entities)
        random.shuffle(generated_entities)
        generated_entities = generated_entities[:num_entities]
        return [str(entity).strip() for entity in generated_entities]

    def generate_demos_fixed_one(self):
        """
        Generate demonstrations with sentences containing one entity for each entity type.
        :return:
        """
        def _generate_sentence(entity_type, entity_mention):

            query = (f"Create a sentence containing '{entity_mention}', which is an entity of type '{entity_type}'. \n"
                     f"Please do not reveal the type of entity in the sentence")
            inputs = {'query': query}

            res_message = fu.request_dify_completion(
                base_url=self.config.dify_api.base_url,
                inputs=inputs,
                token=self.config.dify_api[self.config.sentence_app.name],
                response_mode="streaming",
            )
            sentence = fu.clean_format(res_message)
            return sentence
        demonstrations = []
        for entity_type, entities in self.entity_list.items():
            for entity in entities:
                generated_sentence = _generate_sentence(entity_type, entity)
                logger.info(f"Generated Sentence: {generated_sentence}")
                output = f'[("{entity_type}", "{entity}")]'
                demonstrations.append(f"sentence: {generated_sentence}\noutput: {output}")

                # todo, Diversify entities ()
                # new_sentence = self._diversify_entity(generated_sentence, entity_type)
                # logger.info(f"New Sentence: {new_sentence}")

                # extract new entities and update entities list
                # new_entity = self._extract_entity(new_sentence, entity_type)
                # if new_entity and new_entity not in self.entity_list[entity_type]:
                #     self.entity_list[entity_type].append(new_entity)
                #     logger.info(f"Added new entity: {new_entity}")
        return demonstrations

    def generate_demos_combined(self):
        """
        Generate demonstrations with sentences containing multiple entities from different entity type.
        :param max_entities: the maximum number of entities in one sentence
        :return:
        """
        assert 'max_entities' in self.config, "'max_entities' should be in the config"
        max_entities = self.config.max_entities
        assert max_entities in range(1, 6), f"max_entities should be in range(1, 4), but got {max_entities}"

        def _generate_sentence(entity_combination):
            """
            :param self:
            :param entity_combination: List[typle(str, str)], [(mention 0 , label 0 ), (mention 1, label 1), ...]
            :return:
            """
            entity_str = ''
            for e, label in entity_combination:
                entity_str += f"{e} (a {label} named entity or phrase), "
            query = (f"Create a sentence containing '{entity_str}',. \n"
                     f"Please do not reveal the type of entity in the sentence")
            inputs = {'query': query}

            res_message = fu.request_dify_completion(
                base_url=self.config.dify_api.base_url,
                inputs=inputs,
                token=self.config.dify_api[self.config.sentence_app.name],
                response_mode="streaming",
            )
            sentence = fu.clean_format(res_message)
            return sentence

        def _get_entity_combinations(original_entity_list: dict):
            # 1. mix all entities with different types
            entity_list, entity_ids = [], []
            idx = 0
            for entity_type, entities in original_entity_list.items():
                for entity in entities:
                    entity_list.append((entity, entity_type))
                    entity_ids.append(idx)
                    idx += 1
            random.shuffle(entity_ids)

            # 2. Get entity combinations with a maximum length of 3
            start = 0
            random_lens = list(range(1, self.config.max_entities + 1))
            while start < len(entity_ids):
                random_len = random.choice(random_lens)  # choose a random length from 1 to max_entities
                end = start + random_len
                yield [entity_list[i] for i in entity_ids[start:end]]
                start = end

        demonstrations = []
        # get entity combinations
        entity_combinations =  _get_entity_combinations(self.entity_list)
        for entity_combination in entity_combinations:
            generated_sentence = _generate_sentence(entity_combination)
            logger.info(f"Generated Sentence: {generated_sentence}")
            output = '['
            for entity, entity_type in entity_combination:
                output += f'("{entity_type}", "{entity}"), '
            output += ']'
            demonstrations.append(f"sentence: {generated_sentence}\noutput: {output}")

        # todo, Diversify entities ()
        # new_sentence = self._diversify_entity(generated_sentence, entity_type)
        # logger.info(f"New Sentence: {new_sentence}")

        # extract new entities and update entities list
        # new_entity = self._extract_entity(new_sentence, entity_type)
        # if new_entity and new_entity not in self.entity_list[entity_type]:
        #     self.entity_list[entity_type].append(new_entity)
        #     logger.info(f"Added new entity: {new_entity}")
        return demonstrations


    # def _diversify_entity(self, sentence, entity_type):
    #     prompt = [
    #         {"role": "system", "content": "You are a helpful assistant"},
    #         {"role": "user",
    #          "content": f"Replace the entity in the following sentence with a new entity of type {entity_type}: {sentence}"}
    #     ]
    #     outputs = self.annotator.llm.chat(messages=prompt, sampling_params=self.annotator.sampling_params,
    #                                       use_tqdm=False)
    #     return outputs[0].outputs[0].text

    def generate_demonstrations(self, generate_method="fixed"):
        """
        Generate demonstrations
        :param generate_method: the method to generate sentence of demonstrations
        :return:
        """
        sentence_generate_methods = {
            "fixed": self.generate_demos_fixed_one,
            "combined": self.generate_demos_combined
        }
        assert generate_method in sentence_generate_methods, (f"generate_method should be in {sentence_generate_methods.keys()}"
                                                              f"But got {generate_method}")

        demonstrations_file = fu.init_file_path(
            config=self.config,
            file_dir=self.config.cache_dir,
            file_postfix_name=f'demonstrations.json'
        )
        if os.path.exists(demonstrations_file):
            with open(demonstrations_file, "r", encoding="utf-8") as f:
                demonstrations = json.load(f)
            logger.info(f"Cached demonstrations found! Load demonstrations from {demonstrations_file}")
            return demonstrations

        # no cached file. generate demonstrations from scratch
        demonstrations = sentence_generate_methods[generate_method]()

        # cache demonstrations to file
        try:
            with open(demonstrations_file, "w", encoding="utf-8") as f:
                json.dump(demonstrations, f, ensure_ascii=False, indent=4)
            logger.info(f"Save demonstrations to {demonstrations_file}")
        except Exception as e:
            logger.error(f"Error during saving demonstrations: {e}")
        return demonstrations

    def get_prompt(self, setting='few-shot'):
        """
        Get the prompt for the user
        :param setting: the setting of the annotation, 'few-shot' or 'zero-shot'
        :return:
        """
        assert setting in {'few-shot', 'zero-shot'}, f"Invalid setting: {setting}, should be one of {'few-shot', 'zero-shot'}"

        demonstrations_str = ''
        if setting != 'zero-shot':
            # step1, generate initial entities
            self.generate_initial_entities()

            # step2, generate demonstrations
            demonstrations = self.generate_demonstrations(self.config.generate_method)
            if (self.config.generate_method == 'combined' and 'augment_method' in self.config
                    and self.config.augment_method == 'lsp'):
                label_subsets = fu.get_label_subsets(
                    self.all_labels,
                    self.subset_size,
                    self.config.partition_time
                )
                idx = 0
                for demon in demonstrations:
                    sentence, output = demon.split('\noutput: ')
                    spans_labels = ast.literal_eval(output)
                    for label_subset in label_subsets:
                        tmp_output = '['
                        for label, mention in spans_labels:
                            if label in label_subset:
                                tmp_output += f'("{label}", "{mention}"),'
                        tmp_output += ']'
                        if tmp_output == '[]':
                            continue
                        demonstrations_str += f'{idx + 1})\n {sentence}\noutput: {tmp_output}\n'
                        idx += 1
            else:
                for idx, demon in enumerate(demonstrations):
                    demonstrations_str += f'{idx + 1})\n {demon} \n'

        # get type information
        types_information = ''
        for idx, (type_str, type_description) in enumerate(self.label_description.items()):
            description = self._get_type_description(type_str)
            types_information += '{idx}) {type}\n {description}\n'.format(idx=idx + 1, type=type_str, description=description)

        # step3, get the prompt
        chat_template_file = fu.init_file_path(
            config=self.config,
            file_dir=self.config.cache_dir,
            file_postfix_name=f'template.txt'
        )
        if os.path.exists(chat_template_file):
            with open(chat_template_file, "r", encoding="utf-8") as f:
                prompt = f.read()
            logger.info(f"Cached template found! Load template from {chat_template_file}")
            return prompt, types_information, demonstrations_str

        # no cached file. generate template from scratch
        prompt = f"""
        ### Role
        You are a professional and helpful crowdsourcing data annotator using English  with the help of description of types.
        ### Task
        1. Identify the entities and recognize their types in the user'e query with the help of description of types.
        Your output should be a string in the format of the tuple list,  like'[(type 0, entity 0), (type 1, entity 1), ...]'.
        2. There are some demonstrations after '### Demonstrations' to help you understand the task better.
        Please note that the demonstrations are generated by yourself and there may be errors in the output. 
        So you cannot directly output your answer based on the output in the demonstrations.
        3. Those demonstrations can only be referenced when you judge them to be correct on your own.
        
        ### Type description
        {types_information}

        """
        if setting == 'few-shot':
            prompt += f"""
            ### Demonstrations
            {demonstrations_str}
            
            """

        try:
            with open(chat_template_file, "w", encoding="utf-8") as f:
                f.write(prompt)
            logger.info(f"Save template to {chat_template_file}")
        except Exception as e:
            logger.error(f"Error during saving template: {e}")
        return prompt, types_information, demonstrations_str

    def generate_entity_for_distribution(self):
        """
        generate entities for statistic distribution of hypernyms
        :return:
        """
        # 1. load cached entities
        entities_file = os.path.join(
            self.config.cache_dir,
            self.config.entity_app.name,
            f'{self.config.dataset.dataset_name}_generated_entities.json'
        )
        logger.info(f'entities_file: {entities_file}')
        if os.path.exists(entities_file):
            try:
                with open(entities_file, "r", encoding="utf-8") as f:
                    entity_list = json.load(f)
                logger.info(f"Cached entities found! Load entities from {entities_file}")
                return entity_list
            except Exception as e:
                logger.error(f"Error during loading entities: {e}")

        # 2. no cached file generate entities from scratch
        entity_list = {entity_type: [] for entity_type in self.entity_types}
        for entity_type in self.entity_types:
            entities = self._generate_entities(entity_type, self.config.num_entities, retry_num=20)
            entity_list[entity_type].extend(entities)
        logger.info("Entity List:")
        logger.info(entity_list)

        # 3. save entities to file
        try:
            with open(entities_file, "w", encoding="utf-8") as f:
                json.dump(entity_list, f, ensure_ascii=False, indent=4)
            logger.info(f"Save generated entities for distribution to {entities_file}")
        except Exception as e:
            logger.error(f"Error during saving generated entities for distribution: {e}")

        return entity_list

    def load_entity_for_distribution(self, dataset, entity_num=100):
        """
        Load the entities for statistic hypernyms distribution
        :param dataset: the dataset to load entities
        :param entity_num: the number of entities to load
        :return:
        """
        # 1. load cached entities
        entities_file = os.path.join(
            self.config.cache_dir, # different entity app use same cache dir to save loaded entities
            'load_entities',
            f'{self.config.dataset.dataset_name}_load_entities.json'
        )
        logger.info(f'entities_file: {entities_file}')
        if os.path.exists(entities_file):
            try:
                with open(entities_file, "r", encoding="utf-8") as f:
                    entity_list = json.load(f)
                logger.info(f"Cached entities found! Load entities from {entities_file}")
                return entity_list
            except Exception as e:
                logger.error(f"Error during loading entities: {e}")

        # 2. no cached file. load entities from scratch
        entity_list = {entity_type: [] for entity_type in self.entity_types}
        for span_label in dataset['spans_labels']:
            for start, end, mention, label in span_label:
                entity_list[label].append(mention)
        for entity_type in self.entity_types:
            random.shuffle(entity_list[entity_type])
            entity_list[entity_type] = entity_list[entity_type][:entity_num]

        # 3. save entities to file
        try:
            with open(entities_file, "w", encoding="utf-8") as f:
                json.dump(entity_list, f, ensure_ascii=False, indent=4)
            logger.info(f"Save loaded entities for distribution to {entities_file}")
        except Exception as e:
            logger.info(f"Error during saving loaded entities for distribution: {e}")

        return entity_list

    def get_entity_hypernyms(self, entities, entity_source='generate'):
        """
        Get the hypernyms (concepts and category) of entities
        :param entities: the entities need to get hypernyms
        :param entity_source: the source of entities, 'generate' or 'load'
        :return:
        """
        # 1. load cached hypernyms
        if entity_source == 'load':
            cache_dir = os.path.join(self.config.cache_dir, 'load_entities')
        else:
            cache_dir = os.path.join(self.config.cache_dir, self.config.entity_app.name)
        hypernyms_file = os.path.join(cache_dir, f'{self.config.dataset.dataset_name}_{entity_source}_hypernyms.json')
        logger.info(f'hypernyms_file: {hypernyms_file}')
        if os.path.exists(hypernyms_file):
            try:
                with open(hypernyms_file, "r", encoding="utf-8") as f:
                    hypernyms = json.load(f)
                logger.info(f"Cached hypernyms found! Load hypernyms from {hypernyms_file}")
                return hypernyms
            except Exception as e:
                logger.error(f"Error during loading hypernyms: {e}")

        # 2. no cached file. generate hypernyms from scratch
        hypernyms = {}
        for entity_type in self.entity_types:
            hypernyms[entity_type] = []
            for batched_entities in tqdm(itertools.batched(entities[entity_type], 2), total=len(entities[entity_type]) // 2):
                retry_num = 10
                while retry_num > 0:
                    inputs = {}
                    query = f"{batched_entities}"
                    response = fu.request_dify_chat(
                        base_url=self.config.dify_api.base_url,
                        token=self.config.dify_api[self.config.hypernyms_app.name],
                        query=query,
                        inputs=inputs,
                        response_mode="streaming",
                        app_mode=self.config.hypernyms_app.app_mode
                    )
                    response = fu.clean_format(response)

                    # extract hypernyms (a json string) from response
                    response = response.replace('\'', '\"')
                    pattern = r'\{.*\}'
                    matches = re.findall(pattern, response, re.DOTALL)
                    if len(matches) == 0:
                        logger.error(f"No match. Error during extracting hypernyms from: {response}")
                        retry_num -= 1
                        continue
                    try:
                        batched_responses = json.loads(matches[0])
                        hypernyms[entity_type].extend(batched_responses['entities'])
                        retry_num = 0  # if no error, break the retry loop
                    except Exception as e:
                        logger.error(f"Error during loading hypernyms: {e}. "
                                     f'Response: {response} \n')
                        retry_num -= 1

        # 3. cache hypernyms to file
        try:
            with open(hypernyms_file, "w", encoding="utf-8") as f:
                json.dump(hypernyms, f, ensure_ascii=False, indent=4)
            logger.info(f"Save hypernyms to {hypernyms_file}")
        except Exception as e:
            logger.error(f"Error during saving hypernyms: {e}")
        return hypernyms

    def get_entity_hypernyms_dist(self, dataset=None, entity_source='generate'):
        """
        Get the hypernyms distribution of entities
        :param dataset: the dataset to get entities
        :param entity_source: the source of entities, 'generate' or 'load'
        :return:
        """
        def process_hypernyms(text):
            if type(text) == str:
                hypernyms = text.split('/')
            elif hasattr(text, '__iter__'):
                hypernyms = text
            for idx, hypernym in enumerate(hypernyms):
                if 'Requires' in hypernym or 'Uncertain' in hypernym or 'Unknown' in hypernym:
                    # 'Requires': require more information or search
                    # 'Uncertain': uncertain about the hypernym
                    # 'Unknown': unknown hypernym
                    hypernyms[idx] = ''
            return hypernyms

        assert entity_source in {'generate', 'load'}, f"Invalid entity_source: {entity_source}, should be one of {'generate', 'load'}"
        if entity_source == 'load':
            assert dataset is not None, "dataset should be provided when entity_source is 'load'"

        # 1. load cached distribution
        if entity_source == 'load':
            cache_dir = os.path.join(self.config.cache_dir, 'load_entities')
        else:
            cache_dir = os.path.join(self.config.cache_dir, self.config.entity_app.name)
        entity_hypernyms_dist_file = os.path.join(
            cache_dir,  f'{self.config.dataset.dataset_name}_{entity_source}_distribution.json'
        )

        logger.info(f'entity_hypernyms_dist_file: {entity_hypernyms_dist_file}')
        if os.path.exists(entity_hypernyms_dist_file):
            try:
                with open(entity_hypernyms_dist_file, "r", encoding="utf-8") as f:
                    entity_hypernyms_dist = json.load(f)
                logger.info(f"Cached entity distribution found! Load entity distribution from {entity_hypernyms_dist_file}")
                return entity_hypernyms_dist
            except Exception as e:
                logger.error(f"Error during loading entity distribution: {e}")

        # 2. generate entity list
        if entity_source == 'generate':
            entities = self.generate_entity_for_distribution()
        else:
            entities = self.load_entity_for_distribution(dataset, entity_num=self.config.num_entities)

        # 3. get hypernyms of entities
        entity_semantic_desc = self.get_entity_hypernyms(entities, entity_source)

        # 4. get flatten hypernyms
        flatten_semantic_desc = {}
        for entity_type in self.entity_types:
            flatten_semantic_desc[entity_type] = []
            for each_entity in entity_semantic_desc[entity_type]:
                if 'concept' in each_entity and each_entity['concept'] is not None:
                    concepts = process_hypernyms(each_entity['concept'])
                    flatten_semantic_desc[entity_type].extend(concepts)
                else:
                    flatten_semantic_desc[entity_type].append('')
                if 'category' in each_entity and each_entity['category'] is not None:
                    categories = process_hypernyms(each_entity['category'])
                    flatten_semantic_desc[entity_type].extend(categories)
                else:
                    flatten_semantic_desc[entity_type].append('')

        # 5. get hypernyms distribution
        entity_hypernyms_dist = {entity_type: Counter(flatten_semantic_desc[entity_type]) for entity_type in self.entity_types}

        # 6. cache distribution to file
        try:
            with open(entity_hypernyms_dist_file, "w", encoding="utf-8") as f:
                json.dump(entity_hypernyms_dist, f, ensure_ascii=False, indent=4)
            logger.info(f"Save entity distribution to {entity_hypernyms_dist_file}")
        except Exception as e:
            logger.error(f"Error during saving entity distribution: {e}")
        return entity_hypernyms_dist

    def calculate_sim_hypernyms_dist(self, dataset):
        """
        Calculate the similarity score between two hypernyms distributions.
        one is the generated one, the other is the loaded one
        :return:
        """
        # 1. load cached similarity hypernyms
        sim_hypernyms_file = os.path.join(
            self.config.eval_dir,
            self.config.entity_app.name,
            f'{self.config.dataset.dataset_name}_similarity_hypernyms.json')
        logger.info(f'sim_hypernyms_file: {sim_hypernyms_file}')
        if os.path.exists(sim_hypernyms_file):
            try:
                with open(sim_hypernyms_file, "r", encoding="utf-8") as f:
                    sim_hypernyms = json.load(f)
                logger.info(f"Cached similarity hypernyms found! Load similarity hypernyms from {sim_hypernyms_file}")
                return sim_hypernyms['hypernyms_similarity']
            except Exception as e:
                logger.error(f"Error during loading similarity hypernyms: {e}")

        # 2. no cached file. calculate similarity from scratch
        # 2.1 load generated hypernyms distribution
        generated_hypernyms_dist = self.get_entity_hypernyms_dist(entity_source='generate')

        # 2.2 load loaded hypernyms distribution
        loaded_hypernyms_dist = self.get_entity_hypernyms_dist(dataset=dataset, entity_source='load')

        # 2.3 merge hypernyms distribution from different entity types
        merged_gen_hypernyms_dist, merged_load_hypernyms_dist = dict(), dict()
        for entity_type in self.entity_types:
            for key, value in generated_hypernyms_dist[entity_type].items():
                if key in merged_gen_hypernyms_dist:
                    merged_gen_hypernyms_dist[key] += value
                else:
                    merged_gen_hypernyms_dist[key] = value
            for key, value in loaded_hypernyms_dist[entity_type].items():
                if key in merged_load_hypernyms_dist:
                    merged_load_hypernyms_dist[key] += value
                else:
                    merged_load_hypernyms_dist[key] = value
        # 2.4 transform to Counter (multiset)
        merged_gen_hypernyms_dist = Counter(merged_gen_hypernyms_dist)
        merged_load_hypernyms_dist = Counter(merged_load_hypernyms_dist)

        # 2.5 calculate similarity score
        intersection = merged_gen_hypernyms_dist & merged_load_hypernyms_dist  # for multiset,  intersection = min(c[x], d[x]) for each x in c and d
        union = merged_gen_hypernyms_dist | merged_load_hypernyms_dist  # for multiset, union = max(c[x], d[x]) for each x in c and d
        sim_score = sum(intersection.values()) / sum(union.values())

        # 3. cache similarity to file
        sim_hypernyms = {'hypernyms_similarity': sim_score}
        try:
            with open(sim_hypernyms_file, "w", encoding="utf-8") as f:
                json.dump(sim_hypernyms, f, ensure_ascii=False, indent=4)
            logger.info(f"Save similarity hypernyms to {sim_hypernyms_file}")
        except Exception as e:
            logger.error(f"Error during saving similarity hypernyms: {e}")
        return sim_score