import os
import json

class NERDemoGenerator:
    def __init__(self, annotator, config):
        self.config = config
        self.annotator = annotator

        # initialize entity list
        self.entity_list = {entity_type: [] for entity_type in self.config['entity_types']}

    def generate_initial_entities(self):
        init_entities_file = os.path.join(self.config.cache_dir, 'initial_entities.json')
        if os.path.exists(init_entities_file):
            with open(init_entities_file, "r", encoding="utf-8") as f:
                self.entity_list = json.load(f)
            logger(f"Cached initial entities found! Load initial entities from {init_entities_file}")
            return

        # no cached file. generate initial entities from scratch
        for entity_type in self.config['entity_types']:
            initial_entities = self._generate_entities(entity_type, self.config['num_initial_entities'])
            self.entity_list[entity_type].extend(initial_entities)
        logger.info("Initial Entity List:")
        logger.info(self.entity_list)

        # save initial entities to file
        try:
            with open(init_entities_file, "w", encoding="utf-8") as f:
                json.dump(self.entity_list, init_entities_file, ensure_ascii=False, indent=4)
            print(f"Save initial entities to {init_entities_file}")
        except Exception as e:
            print(f"Error during saving initial entities: {e}")

    def _generate_entities(self, entity_type, num_entities):
        prompt = [
            {"role": "system", "content": "You are a helpful assistant"},
            # todo，是否添加解释，做消融实验
            {"role": "user",
             "content": f"Generate {num_entities} examples of {entity_type} entities. Separate with commas (,)."}
        ]
        outputs = self.annotator.chat(conversation=prompt, sampling_params=self.annotator.sampling_params,
                                      use_tqdm=True)
        generated_entities = outputs[0].outputs[0].text.split(",")
        return [entity.strip() for entity in generated_entities]

    def generate_demos_fixed_one(self):
        """
        Generate demonstrations with sentences containing one entity for each entity type.
        :return:
        """
        demonstrations = []
        for entity_type, entities in self.entity_list.items():
            for entity in entities:
                generated_sentence = self._generate_sentence(entity_type, entity)
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

    def _generate_sentence(self, entity_type, entity_mention):
        # entity_str = f"<{entity_type}>('{entity_mention}')"
        prompt = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user",
             "content": f"Create a sentence containing '{entity_mention}', which is an entity of type '{entity_type}'."}
        ]
        outputs = self.annotator.chat(conversation=prompt, sampling_params=self.annotator.sampling_params,
                                      use_tqdm=False)
        return outputs[0].outputs[0].text

    def _diversify_entity(self, sentence, entity_type):
        prompt = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user",
             "content": f"Replace the entity in the following sentence with a new entity of type {entity_type}: {sentence}"}
        ]
        outputs = self.annotator.chat(conversation=prompt, sampling_params=self.annotator.sampling_params,
                                      use_tqdm=False)
        return outputs[0].outputs[0].text

    def _extract_entity(self, sentence, entity_type):
        pattern = re.compile(rf"<{entity_type}>\('([^']+)'\)")
        match = pattern.search(sentence)
        if match:
            return match.group(1)
        return None

    def generate_demonstrations(self, generate_method="fixed"):
        demonstrations_file = os.path.join(self.config.cache_dir, 'demonstrations.json')
        if os.path.exists(demonstrations_file):
            with open(demonstrations_file, "r", encoding="utf-8") as f:
                demonstrations = json.load(f)
            logger.info(f"Cached demonstrations found! Load demonstrations from {demonstrations_file}")
            return demonstrations

        sentence_generate_methods = {
            "fixed": self.generate_demos_fixed_one,
        }
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

    def get_prompt(self, ):
        chat_template_file = os.path.join(self.config.cache_dir, f'{self.config.generate_method}_template.txt')
        if os.path.exists(chat_template_file):
            with open(chat_template_file, "r", encoding="utf-8") as f:
                template = f.read()
            logger.info(f"Cached template found! Load template from {chat_template_file}")
            return template

        # no cached file. generate template from scratch
        self.generate_initial_entities()  # step1, generate initial entities
        demonstrations = self.generate_demonstrations(self.config.generate_method)  # step2, generate demonstrations

        # get type information
        types_information = ''
        for label_id, label_info in enumerate(self.config.dataset.labels):
            if self.config.natural_form == 'natural':  # use natual format
                type_string = label_info['natural']
            else:
                type_string = label_info.keys()[0]

            if self.config.des_format == 'empty':  # don't show the description of the types
                description = ''
            else:
                if self.config.des_format == 'simple':  # use simple description
                    description = label_info['description'].split('.')[:2]  # only show the first two sentences
                    description = ' '.join(description)
                else:  # use full description
                    description = label_info['description']

                description = type_string + ' ' + description
            types_information += '{idx}) {type}\n {description}\n'.format(idx=label_id + 1, type=type_string,
                                                                          description=description)

        prompt = f"""
        You are a professional and helpful crowdsourcing data annotator using English.

        Here is your task:
        ### Task
        Identify the entities and recognize their types in the sentence.
        The output should be a string in the format of the tuple list,  like'[(type 0, entity 0), (type 1, entity 1), ...]'.

        Given types : 
        ### Types
        {types_information}

        Here are some demonstrations to help you understand the task better:
        ### Demonstrations
        {demonstrations}

        """

        try:
            with open(chat_template_file, "w", encoding="utf-8") as f:
                f.write(prompt)
            logger.info(f"Save template to {chat_template_file}")
        except Exception as e:
            logger.error(f"Error during saving template: {e}")
        return prompt