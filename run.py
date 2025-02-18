import os
import re
import hydra
from omegaconf import DictConfig, OmegaConf

from module import Annotator, NERDemoGeneratorOnline, NERDemoGeneratorOffline
from module import func_util as fu

logger = fu.get_logger('run')

@hydra.main(version_base=None, config_path="./cfgs", config_name="config")
def main(cfg: DictConfig):
    # 1. load annotator for all experiments
    if not cfg.online:
        annotator = Annotator(annotator_cfg=cfg.annotator, api_cfg=None)

    k_shots = [1, 5]

    seeds = [22, 32, 42]
    # 2. load and process different datasets
    cache_dir = cfg.cache_dir  # backup
    eval_dir = cfg.eval_dir
    for dataset_name, dataset_cfg_path in cfg.data_cfg_paths.items():
        for k_shot in k_shots:
            for seed in seeds:
                cfg.dataset = OmegaConf.load(dataset_cfg_path)
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

                if cfg.online:
                    demon_generator = NERDemoGeneratorOnline(cfg)
                else:
                    demon_generator = NERDemoGeneratorOffline(annotator, cfg)
                prompt = demon_generator.get_prompt()

if __name__ == "__main__":
    main()
