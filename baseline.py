import re
import hydra
from omegaconf import DictConfig, OmegaConf

from module import Annotator, NERDemoGenerator
from module import func_util as fu

logger = fu.get_logger('run')


@hydra.main(version_base=None, config_path="./cfgs", config_name="config")
def main(cfg: DictConfig):
    fu.set_seed(cfg.seed)
    cfg.cache_dir = cfg.cache_dir.format(
        annotator=cfg.annotator.name,
        dataset=cfg.dataset.dataset_name,
        seed=cfg.seed
    )  # save annotation.py results
    cfg.eval_dir = cfg.eval_dir.format(
        annotator=cfg.annotator.name,
        dataset=cfg.dataset.dataset_name,
        seed=cfg.seed
    )  # save evaluation results
    logger.info("*" * 20 + "Running Args" + "*" * 20)
    logger.info(OmegaConf.to_yaml(cfg))
    logger.info("*" * 50)

    annotator = Annotator(annotator_cfg=cfg.annotator, api_cfg=None)
    demon_generator = NERDemoGenerator(annotator, cfg)
    prompt = demon_generator.get_prompt()

if __name__ == "__main__":
    main()
