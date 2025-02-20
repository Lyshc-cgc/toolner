import ast
import re
import os
import json
import random
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
        logger.info(":Initial Entity List")
        logger.info(self.entity_list)

        # save initial entities to file
        try:
            with open(init_entities_file, "w", encoding="utf-8") as f:
                json.dump(self.entity_list, f, ensure_ascii=False, indent=4)
            print(f"Save initial entities to {init_entities_file}")
        except Exception as e:
            print(f"Error during saving initial entities: {e}")

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

    def _generate_entities(self, entity_type, num_entities):
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
                return None

        type_description = self._get_type_description(entity_type)
        inputs = {'type_description': type_description}
        query = f'generate {num_entities + 3} "{entity_type}" named entities or phrases. '  # 3 more to avoid the same or common entities

        generated_entities = None
        while generated_entities is None:
            res_message= fu.request_dify_chat(
                base_url=self.config.dify_api.base_url,
                token=self.config.dify_api[self.config.entity_app.name],
                query=query,
                inputs=inputs,
                response_mode="streaming",
                app_mode=self.config.entity_app.app_mode
            )

            res_message = fu.clean_format(res_message)
            generated_entities = _extract_entity(res_message)
        return [entity.strip() for entity in generated_entities]

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
                # print(f"New Sentence: {new_sentence}")

                # extract new entities and update entities list
                # new_entity = self._extract_entity(new_sentence, entity_type)
                # if new_entity and new_entity not in self.entity_list[entity_type]:
                #     self.entity_list[entity_type].append(new_entity)
                #     print(f"Added new entity: {new_entity}")
        return demonstrations

    def generate_demos_combined(self):
        """
        Generate demonstrations with sentences containing multiple entities from different entity type.
        :param max_entities: the maximum number of entities in one sentence
        :return:
        """
        assert 'max_entities' in self.config, "'max_entities' should be in the config"
        max_entities = self.config.max_entities
        assert max_entities in range(1, 4), f"max_entities should be in range(1, 4), but got {max_entities}"

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
        # print(f"New Sentence: {new_sentence}")

        # extract new entities and update entities list
        # new_entity = self._extract_entity(new_sentence, entity_type)
        # if new_entity and new_entity not in self.entity_list[entity_type]:
        #     self.entity_list[entity_type].append(new_entity)
        #     print(f"Added new entity: {new_entity}")
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
        assert generate_method in sentence_generate_methods, f"generate_method should be in {sentence_generate_methods.keys()}"

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
            # todo, 这里可以加入lsp方法，
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
        You are a professional and helpful crowdsourcing data annotator using English.

        Here is your task:
        ### Task
        Identify the entities and recognize their types in the user's query.
        The output should be a string in the format of the tuple list,  like'[(type 0, entity 0), (type 1, entity 1), ...]'.

        Given types : 
        ### Types
        {types_information}

        """
        if setting == 'few-shot':
            prompt += f"""
            Here are some demonstrations to help you understand the task better:
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
