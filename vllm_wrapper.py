import copy
import os 
from vllm import LLM
from vllm.sampling_params import SamplingParams
from modelscope import AutoTokenizer, GenerationConfig

# 通义千问的特殊token
IMSTART='<|im_start|>'  
IMEND='<|im_end|>'
ENDOFTEXT='<|endoftext|>'     # EOS以及PAD都是它

class vLLMWrapper:
    def __init__(self, 
                 model_dir,
                 tensor_parallel_size=1,
                 gpu_memory_utilization=0.90,
                 dtype='float16',
                 quantization=None):
        # 模型目录下的generation_config.json文件，是推理的关键参数
        '''
        {
            "chat_format": "chatml",
            "eos_token_id": 151643,
            "pad_token_id": 151643,
            "max_window_size": 6144,
            "max_new_tokens": 512,
            "do_sample": true,
            "top_k": 0,
            "top_p": 0.8,
            "repetition_penalty": 1.1,
            "transformers_version": "4.31.0"
            }
        '''
        self.generation_config = GenerationConfig.from_pretrained(model_dir,trust_remote_code=True)
        
        # 加载分词器
        self.tokenizer=AutoTokenizer.from_pretrained(model_dir,trust_remote_code=True)
        self.tokenizer.eos_token_id=self.generation_config.eos_token_id
        # 推理终止词，遇到这些词停止继续推理
        self.stop_words_ids=[self.tokenizer.im_start_id,self.tokenizer.im_end_id,self.tokenizer.eos_token_id]
        # vLLM加载模型
        os.environ['VLLM_USE_MODELSCOPE']='True'
        self.model=LLM(model=model_dir,
                       tokenizer=model_dir,
                       tensor_parallel_size=tensor_parallel_size,
                       trust_remote_code=True,
                       quantization=quantization,
                       gpu_memory_utilization=gpu_memory_utilization,
                       dtype=dtype)
    
    # 按chatml格式构造千问的Prompt
    def build_prompt(self,
                     tokenizer,
                     query,
                     history=None,
                     system=""):
        if history is None:
            history=[]

        # 包裹发言内容的token
        im_start,im_start_tokens='<|im_start|>',[tokenizer.im_start_id]
        im_end,im_end_tokens='<|im_end|>',[tokenizer.im_end_id]
        # 换行符token
        nl_tokens=tokenizer.encode("\n")

        # 用于编码system/user/assistant的一段发言, 格式{role}\n{content}
        def _tokenize_str(role,content): # 返回元组，下标0是文本，下标1是token ids
            return f"{role}\n{content}",tokenizer.encode(role)+nl_tokens+tokenizer.encode(content)
        
        # 剩余token数
        left_token_space=self.generation_config.max_window_size

        # prompt头部: system发言
        system_text_part,system_tokens_part=_tokenize_str("system", system) # system_tokens_part -->    system\nYou are a helpful assistant.
        system_text=f'{im_start}{system_text_part}{im_end}'
        system_tokens=im_start_tokens+system_tokens_part+im_end_tokens # <|im_start|>system\nYou are a helpful assistant.<|im_end|>
        left_token_space-=len(system_tokens)
        
        # prompt尾部: user发言和assistant引导
        query_text_part,query_tokens_part=_tokenize_str('user', query)
        query_tokens_prefix=nl_tokens+ im_start_tokens
        query_tokens_suffix=im_end_tokens+nl_tokens+im_start_tokens+tokenizer.encode('assistant')+nl_tokens
        if len(query_tokens_prefix)+len(query_tokens_part)+len(query_tokens_suffix)>left_token_space: # query太长截断
            query_token_len=left_token_space-len(query_tokens_prefix)-len(query_tokens_suffix)
            query_tokens_part=query_tokens_part[:query_token_len]
            query_text_part=self.tokenizer.decode(query_tokens_part)
        query_tokens=query_tokens_prefix+query_tokens_part+query_tokens_suffix
        query_text=f"\n{im_start}{query_text_part}{im_end}\n{im_start}assistant\n"
        left_token_space-=len(query_tokens)
        
        # prompt腰部: 历史user+assitant对话
        history_text,history_tokens='',[]
        for hist_query,hist_response in reversed(history):    # 优先采用最近的对话历史
            hist_query_text,hist_query_tokens_part=_tokenize_str("user",hist_query) # user\n历史提问
            hist_response_text,hist_response_tokens_part=_tokenize_str("assistant",hist_response) # assistant\n历史回答
            # 生成本轮对话
            cur_history_tokens=nl_tokens+im_start_tokens+hist_query_tokens_part+im_end_tokens+nl_tokens+im_start_tokens+hist_response_tokens_part+im_end_tokens
            cur_history_text=f"\n{im_start}{hist_query_text}{im_end}\n{im_start}{hist_response_text}{im_end}"
            # 储存多轮对话
            if len(cur_history_tokens)<=left_token_space:
                history_text=cur_history_text+history_text
                history_tokens=cur_history_tokens+history_tokens
            else:
                break 
                
        # 生成完整Prompt
        prompt_str=f'{system_text}{history_text}{query_text}'
        prompt_tokens=system_tokens+history_tokens+query_tokens
        return prompt_str,prompt_tokens

    def chat(self,query,history=None,system="You are a helpful assistant.",extra_stop_words_ids=[]):
        # 历史聊天
        if history is None:
            history = []
        else:
            history = copy.deepcopy(history)

        # 额外指定推理停止词
        stop_words_ids=self.stop_words_ids+extra_stop_words_ids

        # 构造prompt
        prompt_text,prompt_tokens=self.build_prompt(self.tokenizer,query,history=history,system=system)
        
        # 打开注释，观测底层Prompt构造
        # print(prompt_text)

        # VLLM请求配置
        sampling_params=SamplingParams(stop_token_ids=stop_words_ids, 
                                         early_stopping=False,
                                         top_p=self.generation_config.top_p,
                                         top_k=-1 if self.generation_config.top_k == 0 else self.generation_config.top_k,
                                         temperature=self.generation_config.temperature,
                                         repetition_penalty=self.generation_config.repetition_penalty,
                                         max_tokens=self.generation_config.max_new_tokens)
        
        # 调用VLLM执行推理（批次大小1）
        req_outputs=self.model.generate(prompt_token_ids=[prompt_tokens],sampling_params=sampling_params)
        req_output=req_outputs[0]    
        
        # transformer模型的原生返回
        response=req_output.outputs[0].text
        # 清楚可能出现的内置终止词
        for token_id in self.stop_words_ids:
            response=response.replace(self.tokenizer.decode(token_id),'')

        # 整理历史对话
        history.append((query,response))
        return response,history