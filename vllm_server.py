import os
from vllm import AsyncEngineArgs, AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from modelscope import AutoTokenizer, GenerationConfig, snapshot_download
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
import uvicorn
from prompt_utils import _build_prompt, remove_stop_words
import uuid
import json

# http接口服务
app = FastAPI()

# vLLM参数
model_dir = "qwen/Qwen-14B-Chat-Int4"
tensor_parallel_size = 1
gpu_memory_utilization = 0.6
quantization = 'gptq'
dtype = 'float16'


# vLLM模型加载
def load_vllm():
    global generation_config, tokenizer, stop_words_ids, engine
    # 模型下载
    snapshot_download(model_dir)
    # 模型基础配置
    generation_config = GenerationConfig.from_pretrained(model_dir, trust_remote_code=True)
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    tokenizer.eos_token_id = generation_config.eos_token_id
    # 推理终止词
    stop_words_ids = [tokenizer.im_start_id, tokenizer.im_end_id, tokenizer.eos_token_id]
    # vLLM基础配置
    args = AsyncEngineArgs(model_dir)
    args.worker_use_ray = False
    args.engine_use_ray = False
    args.tokenizer = model_dir
    args.tensor_parallel_size = tensor_parallel_size
    args.trust_remote_code = True
    args.quantization = quantization
    args.gpu_memory_utilization = gpu_memory_utilization
    args.dtype = dtype
    args.max_num_seqs = 20  # batch最大20条样本
    # 加载模型
    os.environ['VLLM_USE_MODELSCOPE'] = 'True'
    engine = AsyncLLMEngine.from_engine_args(args)
    return generation_config, tokenizer, stop_words_ids, engine


generation_config, tokenizer, stop_words_ids, engine = load_vllm()


# 用户停止句匹配
def match_user_stop_words(response_token_ids, user_stop_tokens):
    for stop_tokens in user_stop_tokens:
        if len(response_token_ids) < len(stop_tokens):
            continue
        if response_token_ids[-len(stop_tokens):] == stop_tokens:
            return True  # 命中停止句, 返回True
    return False


# chat对话接口
@app.post("/chat")
async def chat(request: Request):
    request = await request.json()

    query = request.get('query', None)
    history = request.get('history', [])
    system = request.get('system', 'You are a helpful assistant.')
    stream = request.get("stream", False)
    user_stop_words = request.get("user_stop_words",
                                  [])  # list[str]，用户自定义停止句，例如：['Observation: ', 'Action: ']定义了2个停止句，遇到任何一个都会停止

    if query is None:
        return Response(status_code=502, content='query is empty')

    # 用户停止词
    user_stop_tokens = []
    for words in user_stop_words:
        user_stop_tokens.append(tokenizer.encode(words))

    # 构造prompt
    prompt_text, prompt_tokens = _build_prompt(generation_config, tokenizer, query, history=history, system=system)

    # vLLM请求配置
    sampling_params = SamplingParams(stop_token_ids=stop_words_ids,
                                     early_stopping=False,
                                     top_p=generation_config.top_p,
                                     top_k=-1 if generation_config.top_k == 0 else generation_config.top_k,
                                     temperature=generation_config.temperature,
                                     repetition_penalty=generation_config.repetition_penalty,
                                     max_tokens=generation_config.max_new_tokens)
    # vLLM异步推理（在独立线程中阻塞执行推理，主线程异步等待完成通知）
    request_id = str(uuid.uuid4().hex)
    results_iter = engine.generate(prompt=None, sampling_params=sampling_params, prompt_token_ids=prompt_tokens,
                                   request_id=request_id)

    # 流式返回，即迭代transformer的每一步推理结果并反复返回
    if stream:
        async def streaming_resp():
            async for result in results_iter:
                # 移除im_end,eos等系统停止词
                token_ids = remove_stop_words(result.outputs[0].token_ids, stop_words_ids)
                # 返回截止目前的tokens输出                
                text = tokenizer.decode(token_ids)
                yield (json.dumps({'text': text}) + '\0').encode('utf-8')
                # 匹配用户停止词,终止推理
                if match_user_stop_words(token_ids, user_stop_tokens):
                    await engine.abort(request_id)  # 终止vllm后续推理
                    break

        return StreamingResponse(streaming_resp())

    # 整体一次性返回模式
    async for result in results_iter:
        # 移除im_end,eos等系统停止词
        token_ids = remove_stop_words(result.outputs[0].token_ids, stop_words_ids)
        # 返回截止目前的tokens输出                
        text = tokenizer.decode(token_ids)
        # 匹配用户停止词,终止推理
        if match_user_stop_words(token_ids, user_stop_tokens):
            await engine.abort(request_id)  # 终止vllm后续推理
            break

    ret = {"text": text}
    return JSONResponse(ret)


if __name__ == '__main__':
    uvicorn.run(app,
                host=None,
                port=8000,
                log_level="debug")
