import copy
import os 
from vllm import LLM
from vllm.sampling_params import SamplingParams
from modelscope import AutoTokenizer, GenerationConfig, snapshot_download
from prompt_utils import _build_prompt,remove_stop_words

# 通义千问的特殊token
IMSTART='<|im_start|>'        # 聊天的开始
IMEND='<|im_end|>'            # 聊天的结束   
ENDOFTEXT='<|endoftext|>'     # 整个对话的结束，EOS以及PAD都是它

class vLLMWrapper:
    def __init__(self, 
                model_dir,
                tensor_parallel_size=DEFAULT_TENSOR_PARALLEL_SIZE,
                gpu_memory_utilization=DEFAULT_GPU_MEMORY_UTILIZATION,
                dtype=DEFAULT_DTYPE,
                quantization=None):
            
        # 初始化日志
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        
        # 参数校验
        self.validate_params(tensor_parallel_size, gpu_memory_utilization)
        
        # 设置环境变量
        os.environ['VLLM_USE_MODELSCOPE'] = 'True'
        
        try:
            # 模型下载
            self.snapshot_download(model_dir)
            # 加载生成配置项
            self.generation_config = self.load_generation_config(model_dir)
            # 加载分词器
            self.tokenizer = self.load_tokenizer(model_dir, self.generation_config)
            # 推理终止词
            self.stop_words_ids = [self.tokenizer.im_start_id, self.tokenizer.im_end_id, self.tokenizer.eos_token_id]
            # vLLM加载模型
            self.model = self.load_model(model_dir, 
                                        tensor_parallel_size=tensor_parallel_size,
                                        quantization=quantization,
                                        gpu_memory_utilization=gpu_memory_utilization,
                                        dtype=dtype)
        except Exception as e:
            logging.error(f"Failed to initialize model loader: {e}")
        
    def validate_params(self, tensor_parallel_size, gpu_memory_utilization):
        if tensor_parallel_size <= 0:
            raise ValueError("tensor_parallel_size must be greater than 0")
        if gpu_memory_utilization <= 0 or gpu_memory_utilization > 1:
            raise ValueError("gpu_memory_utilization must be between 0 and 1")
    
    def snapshot_download(self, model_dir):
        logging.info("Downloading model snapshot...")
        # 下载逻辑
        snapshot_download(model_dir)
        logging.info("Model snapshot downloaded.")

    def load_generation_config(self, model_dir):
        logging.info("Loading generation config...")
        config = GenerationConfig.from_pretrained(model_dir, trust_remote_code=False)
        logging.info("Generation config loaded.")
        return config

    def load_tokenizer(self, model_dir, config):
        logging.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=False)
        tokenizer.eos_token_id = config.eos_token_id
        logging.info("Tokenizer loaded.")
        return tokenizer

    def load_model(self, model_dir, tensor_parallel_size, quantization, gpu_memory_utilization, dtype):
        logging.info("Loading LLM model...")
        model = LLM(model=model_dir,
                    tokenizer=model_dir,
                    tensor_parallel_size=tensor_parallel_size,
                    trust_remote_code=False,
                    quantization=quantization,
                    gpu_memory_utilization=gpu_memory_utilization,
                    dtype=dtype)
        logging.info("LLM model loaded.")
        return model

    def chat(self, query, history = None, system = "You are a helpful assistant.", extra_stop_words_ids = []):
        """
        允许用户输入一个查询问题 (query)
        并根据可选的历史交流 (history) 以及定义的系统角色 (system) 生成一个回答
        """
        if history is None:
            history = []
        else:
            history = list(history)  # 使用list()避免深拷贝，这里假设history不会被外部修改

        # 额外指定推理停止词
        stop_words_ids = self.stop_words_ids + extra_stop_words_ids

        # 核心：构造prompt
        prompt_text, prompt_tokens = _build_prompt(self.generation_config, self.tokenizer, query, history=history, system=system)
        
        # VLLM请求配置
        sampling_params = SamplingParams(stop_token_ids=stop_words_ids, 
                                         early_stopping=False,
                                         top_p=self.generation_config.top_p,
                                         top_k=-1 if self.generation_config.top_k == 0 else self.generation_config.top_k,
                                         temperature=self.generation_config.temperature,
                                         repetition_penalty=self.generation_config.repetition_penalty,
                                         max_tokens=self.generation_config.max_new_tokens)
        
        try:
            # 调用VLLM执行推理（批次大小1）
            req_outputs = self.model.generate(prompt_token_ids=[prompt_tokens], sampling_params=sampling_params, use_tqdm=False) # use_tqdm禁止进度条
            req_output = req_outputs[0]    
        
            # 移除停用词        
            response_token_ids = remove_stop_words(req_output.outputs[0].token_ids, stop_words_ids)
            response = self.tokenizer.decode(response_token_ids)
        
            # 整理历史对话
            history.append((query, response))
            return response, history
        except Exception as e:
            # 对可能的错误进行处理，这里简单返回一个错误提示和空历史记录
            print(f"Error during chat: {e}")
            return "An error occurred.", []