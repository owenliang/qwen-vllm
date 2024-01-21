import copy
from typing import List, Optional, Tuple

import torch
from vllm import LLM
from vllm.sampling_params import SamplingParams
from modelscope import AutoTokenizer, GenerationConfig

IMEND = "<|im_end|>"
ENDOFTEXT = "<|endoftext|>"

HistoryType = List[Tuple[str, str]]
TokensType = List[int]
BatchTokensType = List[List[int]]

def get_stop_words_ids(tokenizer):
    stop_words_ids = [[tokenizer.im_end_id],[tokenizer.eos_token_id],[tokenizer.im_start_id]]
    return stop_words_ids

# 构造transformer模型的输入
def make_context(
    tokenizer,
    query,
    history=None,
    system="",
    max_window_size=6144
):
    if history is None:
        history = []

    im_start, im_end = "<|im_start|>", "<|im_end|>"
    im_start_tokens = [tokenizer.im_start_id]
    im_end_tokens = [tokenizer.im_end_id]
    nl_tokens = tokenizer.encode("\n")

    def _tokenize_str(role, content): # 返回元组，下标0是文本，下标1是token ids
        return f"{role}\n{content}",  tokenizer.encode(role, allowed_special=set()) + nl_tokens + tokenizer.encode(content, allowed_special=set())

    system_text, system_tokens_part = _tokenize_str("system", system) # system_tokens_part -->    system\nYou are a helpful assistant.
    system_tokens = im_start_tokens + system_tokens_part + im_end_tokens # <|im_start|>system\nYou are a helpful assistant.<|im_end|>

    raw_text = ""
    context_tokens = []

    for turn_query, turn_response in reversed(history): # 倒序遍历聊天历史
        query_text, query_tokens_part = _tokenize_str("user", turn_query) # query_tokens_part -->   user\n历史提问
        query_tokens = im_start_tokens + query_tokens_part + im_end_tokens  # query_tokens --> <|im_start|>user\n历史提问<|im_end|>
        response_text, response_tokens_part = _tokenize_str( # response_tokens_part -->   assistant\n历史回答
            "assistant", turn_response
        ) 
        response_tokens = im_start_tokens + response_tokens_part + im_end_tokens # response_tokens --> <|im_start|>assistant\n历史回答<|im_end|>

        next_context_tokens = nl_tokens + query_tokens + nl_tokens + response_tokens    # next_context_tokens --> \n<|im_start|>user\n历史提问<|im_end|>\n<|im_start|>assistant\n历史回答<|im_end|>
        prev_chat = (
            f"\n{im_start}{query_text}{im_end}\n{im_start}{response_text}{im_end}"
        )

        # 本轮历史对话
        # 看一下本轮历史对话+累计长度是不是超过最大输入（positional embedding有限制）
        current_context_size = (
            len(system_tokens) + len(next_context_tokens) + len(context_tokens)
        )
        if current_context_size < max_window_size:
            context_tokens = next_context_tokens + context_tokens # 本轮对话放到头部，生成逻辑很清晰，先生成最近的历史对话，再生成稍远的历史对话，直到放不下为止
            raw_text = prev_chat + raw_text # 对应的文本版本
        else:
            break
    # # # # # # # # # # # # # # # 历史对话生成结束
    
    # TODO：要先预留出system和本轮提问的token空间，再去生成history，这段代码要调一下顺序
    
    # 把system加上
    context_tokens = system_tokens + context_tokens # <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n历史提问<|im_end|>\n<|im_start|>assistant\n历史回答<|im_end|>\n<|im_start|>user\n历史提问<|im_end|>\n<|im_start|>assistant\n历史回答<|im_end|>
    raw_text = f"{im_start}{system_text}{im_end}" + raw_text # 对应的文本版本

    # 本轮的提问和引导标签
    # context_tokens --> <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n历史提问<|im_end|>\n<|im_start|>assistant\n历史回答<|im_end|>\n<|im_start|>user\n历史提问<|im_end|>\n<|im_start|>assistant\n历史回答<|im_end|>\n<|im_start|>user\n本轮提问<|im_end|>\n<|im_start|>assistant\n
    context_tokens += (
        nl_tokens # \n
        + im_start_tokens # <|im_start|>
        + _tokenize_str("user", query)[1] # user\n本次提问
        + im_end_tokens # <|im_end|>
        + nl_tokens # \n
        + im_start_tokens # <|im_start|>
        + tokenizer.encode("assistant") # assistant
        + nl_tokens # \n
    )
    # 对应的文本版本，加在历史对话的后面
    raw_text += f"\n{im_start}user\n{query}{im_end}\n{im_start}assistant\n"

    return raw_text, context_tokens

class vLLMWrapper:
    def __init__(self,
               model_dir: str,
               trust_remote_code: bool = True,
               tensor_parallel_size: int = 1,
               gpu_memory_utilization: float = 0.90,
               dtype: str = "bfloat16",
               **kwargs):

        # build generation_config
        self.generation_config = GenerationConfig.from_pretrained(model_dir, trust_remote_code=trust_remote_code)

        # build tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        self.tokenizer.eos_token_id = self.generation_config.eos_token_id

        self.stop_words_ids = []

        quantization = kwargs.get('quantization', None)

        import os 
        os.environ['VLLM_USE_MODELSCOPE']='True'
        
        self.model = LLM(model=model_dir,
                            tokenizer=model_dir,
                            tensor_parallel_size=tensor_parallel_size,
                            trust_remote_code=trust_remote_code,
                            quantization=quantization,
                            gpu_memory_utilization=gpu_memory_utilization,
                            dtype=dtype)

        # 停止继续推理的token
        for stop_id in get_stop_words_ids(self.tokenizer):
            self.stop_words_ids.extend(stop_id)
        self.stop_words_ids.extend([self.generation_config.eos_token_id])

    def chat(self,query,history=None,system="You are a helpful assistant.",**kwargs):
        # 历史聊天
        if history is None:
            history = []
        else:
            history = copy.deepcopy(history)

        # 推理停止词
        extra_stop_words_ids = kwargs.get('stop_words_ids', None)
        if extra_stop_words_ids is None:
            extra_stop_words_ids = []

        max_window_size = kwargs.get('max_window_size', None)
        if max_window_size is None:
            max_window_size = self.generation_config.max_window_size

        # VLLM推理入参
        sampling_params = SamplingParams(stop_token_ids=self.stop_words_ids, # 停止推理的token词
                                            early_stopping=False,
                                            top_p=self.generation_config.top_p,
                                            top_k=-1 if self.generation_config.top_k == 0 else self.generation_config.top_k,
                                            temperature=self.generation_config.temperature,
                                            repetition_penalty=self.generation_config.repetition_penalty,
                                            max_tokens=self.generation_config.max_new_tokens,
                                        )

        # 生成transformer模型输入token ids
        _, context_tokens = make_context(   # context_tokens就是model输入
            self.tokenizer,
            query,
            history=history,
            system=system,
            max_window_size=max_window_size,
        )

        # 作为batch传入 --> prompt_token_ids=[context_tokens]
        # 这里就推理1个提问，其实可以批量推
        req_outputs = self.model.generate([query],
                                            sampling_params=sampling_params,
                                            prompt_token_ids=[context_tokens])
        req_output = req_outputs[0] # 取第1条样本的推理结果

        prompt_str = req_output.prompt
        
        output_str = req_output.outputs[0].text
        output_str=output_str.replace(IMEND,"").replace(ENDOFTEXT,"")

        # 历史对话
        history.append((prompt_str,output_str))
        
        # 本轮返回
        response=output_str

        return response, history