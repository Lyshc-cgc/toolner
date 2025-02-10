import os
from openai import OpenAI
from vllm import LLM, SamplingParams
from module import func_util as fu

logger = fu.get_logger('Annotator')
class Annotator:
    def __init__(self, annotator_cfg, api_cfg, just_test=False):
        """
        Initialize the annotator model.

        :param annotator_cfg: the configuration of the local annotator model.
        :param api_cfg: the configuration of the LLM API.
        :param annotator_cfg:
        :param just_test: If True, we just want to test something in the annotation.py, so that we don't need to load the model.
        """
        self.use_api = True if api_cfg else False
        self.annotator_cfg = annotator_cfg
        self.api_cfg = api_cfg if api_cfg else None

        # 1. GPU setting
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # avoid parallelism in tokenizers
        cuda_devices = [str(i) for i in range(annotator_cfg['tensor_parallel_size'])]
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(cuda_devices)

        # 2. Init the annotating model
        if not just_test:
            logger.info('----- Init LLM -----')
            if self.use_api:
                self.client = OpenAI(
                    api_key=os.getenv(self.api_cfg['api_key']),
                    base_url=self.api_cfg['base_url'],  # base_url
                    max_retries=3,
                )
            else:
                # if not use api, we employ local annotator using vllm
                # https://docs.vllm.ai/en/latest/getting_started/quickstart.html
                self.llm = LLM(model=annotator_cfg['checkpoint'],
                                      tensor_parallel_size=annotator_cfg['tensor_parallel_size'],
                                      dtype=annotator_cfg['dtype'],
                                      gpu_memory_utilization=annotator_cfg['gpu_memory_utilization'],
                                      trust_remote_code=True,
                                      # https://github.com/vllm-project/vllm/issues/6723
                                      # set explicitly enable_chunked_prefill to False For Volta GPU
                                      enable_chunked_prefill=False)
                self.sampling_params = SamplingParams(temperature=annotator_cfg['anno_temperature'],
                                                      top_p=annotator_cfg['anno_top_p'],
                                                      max_tokens=annotator_cfg['anno_max_tokens'],
                                                      repetition_penalty=annotator_cfg['repetition_penalty'])

                # get anno_model's tokenizer to apply the chat template
                # https://github.com/vllm-project/vllm/issues/3119
                # anno_tokenizer = anno_model.llm_engine.tokenizer.tokenizer
                self.anno_tokenizer = self.llm.get_tokenizer()