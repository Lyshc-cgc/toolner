import os
import hydra
import multiprocess
from omegaconf import DictConfig, OmegaConf
from module import NERDemoGenerator, Processor, Annotation
from module import func_util as fu

logger = fu.get_logger('run')
MAX_ENTITIES_NUMS= [3]  # [2, 3, 4, 5], the number of entities in the prompt
seeds = [22, 32, 42]
test_size = 200  # the number of test instances

@hydra.main(version_base=None, config_path="./cfgs", config_name="fs_config")
def run_few_shot(cfg: DictConfig):
    k_shots = [1, 5]
    cache_dir = cfg.cache_dir  # backup
    eval_dir = cfg.eval_dir
    max_entities_nums = [3] if cfg.generate_method == 'fixed' else MAX_ENTITIES_NUMS
    for max_entities in max_entities_nums:
        cfg.max_entities = max_entities
        for dataset_name, dataset_cfg_path in cfg.data_cfg_paths.items():
            # 1. load and process the dataset
            cfg.dataset = OmegaConf.load(dataset_cfg_path)
            processor = Processor(cfg)
            dataset = processor.process()

            for k_shot in k_shots:
                for seed in seeds:
                    # 1. shuffle and select test instances
                    dataset = dataset.shuffle(seed=seed).flatten_indices().select(range(test_size))

                    # 2. init config
                    cfg.k_shot = k_shot
                    cfg.seed = seed
                    cfg.cache_dir = cache_dir.format(
                        entity_app=cfg.entity_app.name,
                        dataset=dataset_name,
                        k_shot=k_shot,
                        seed=seed
                    )  # save annotation.py results
                    cfg.eval_dir = eval_dir.format(
                        entity_app=cfg.entity_app.name,
                        dataset=dataset_name,
                        k_shot=k_shot,
                        seed=seed
                    )  # save evaluation results
                    if not os.path.exists(cfg.cache_dir):
                        os.makedirs(cfg.cache_dir)
                    if not os.path.exists(cfg.eval_dir):
                        os.makedirs(cfg.eval_dir)
                    # logger.info("*" * 20 + " Running Args " + "*" * 20)
                    # logger.info(OmegaConf.to_yaml(cfg))
                    # logger.info("*" * 50)

                    # 3. generate demonstrations and prompt
                    demon_generator = NERDemoGenerator(cfg)
                    prompt, types_information, demonstrations_str = demon_generator.get_prompt(setting='few-shot')

                    # 4. annotate the dataset using few-shot setting
                    annotation = Annotation(cfg)
                    annotation.annotate(
                        dataset=dataset,
                        types_description=types_information,
                        demonstrations=demonstrations_str,
                        setting = 'few-shot'
                    )

@hydra.main(version_base=None, config_path="./cfgs", config_name="0s_config")
def run_zero_shot(cfg: DictConfig):
    cache_dir = cfg.cache_dir  # backup
    eval_dir = cfg.eval_dir
    for dataset_name, dataset_cfg_path in cfg.data_cfg_paths.items():
        # 1. load and process the dataset
        cfg.dataset = OmegaConf.load(dataset_cfg_path)
        processor = Processor(cfg)
        dataset = processor.process()

        for seed in seeds:
            # 1. shuffle and select test instances
            dataset = dataset.shuffle(seed=seed).flatten_indices().select(range(test_size))

            # 2. init config
            cfg.seed = seed
            cfg.cache_dir = cache_dir.format(
                ner_app=cfg.ner_app.name,
                dataset=dataset_name,
                seed=seed
            )  # save annotation.py results
            cfg.eval_dir = eval_dir.format(
                ner_app=cfg.ner_app.name,
                dataset=dataset_name,
                seed=seed
            )  # save evaluation results
            if not os.path.exists(cfg.cache_dir):
                os.makedirs(cfg.cache_dir)
            if not os.path.exists(cfg.eval_dir):
                os.makedirs(cfg.eval_dir)
            # logger.info("*" * 20 + " Running Args " + "*" * 20)
            # logger.info(OmegaConf.to_yaml(cfg))
            # logger.info("*" * 50)

            # 3. generate demonstrations and prompt
            demon_generator = NERDemoGenerator(cfg)
            prompt, types_information, demonstrations_str = demon_generator.get_prompt(setting='zero-shot')

            # 4. annotate the dataset using few-shot setting
            annotation = Annotation(cfg)
            annotation.annotate(
                dataset=dataset,
                types_description=types_information,
                demonstrations=demonstrations_str,
                setting='zero-shot'
            )

if __name__ == "__main__":
    # set 'spawn' start method in the main process to parallelize computation across several GPUs when using multi-processes in the map function
    # refer to https://huggingface.co/docs/datasets/process#map
    multiprocess.set_start_method('spawn')
    # run_few_shot()
    run_zero_shot()