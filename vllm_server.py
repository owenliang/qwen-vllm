import os 
from vllm import AsyncEngineArgs,AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from modelscope import AutoTokenizer, GenerationConfig
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
import uvicorn
from prompt_utils import _build_prompt
import uuid
import json 

# http接口服务
app=FastAPI()

# vLLM参数
model_dir="qwen/Qwen-14B-Chat-Int4"
tensor_parallel_size=1
gpu_memory_utilization=0.6
quantization='gptq'
dtype='float16'

# vLLM模型加载
def load_vllm():
    global generation_config,tokenizer,stop_words_ids,engine    
    
    # 模型基础配置
    generation_config=GenerationConfig.from_pretrained(model_dir,trust_remote_code=True)
    # 加载分词器
    tokenizer=AutoTokenizer.from_pretrained(model_dir,trust_remote_code=True)
    tokenizer.eos_token_id=generation_config.eos_token_id
    # 推理终止词
    stop_words_ids=[tokenizer.im_start_id,tokenizer.im_end_id,tokenizer.eos_token_id]
    # vLLM基础配置
    args=AsyncEngineArgs(model_dir)
    args.worker_use_ray=False
    args.engine_use_ray=False
    args.tokenizer=model_dir
    args.tensor_parallel_size=tensor_parallel_size
    args.trust_remote_code=True
    args.quantization=quantization
    args.gpu_memory_utilization=gpu_memory_utilization
    args.dtype=dtype
    # 加载模型
    os.environ['VLLM_USE_MODELSCOPE']='True'
    engine=AsyncLLMEngine.from_engine_args(args)
    return generation_config,tokenizer,stop_words_ids,engine

generation_config,tokenizer,stop_words_ids,engine=load_vllm()

# chat对话接口
@app.post("/chat")
async def chat(request: Request):
    request=await request.json()
    
    query=request.get('query',None)
    history=request.get('history',[])
    system=request.get('system','You are a helpful assistant.')
    stream=request.get("stream",False)
    if query is None:
        return Response(status_code=502,content='query is empty')
    
    # 构造prompt
    prompt_text,prompt_tokens=_build_prompt(generation_config,tokenizer,query,history=history,system=system)
        
    # vLLM请求配置
    sampling_params=SamplingParams(stop_token_ids=stop_words_ids, 
                                    early_stopping=False,
                                    top_p=generation_config.top_p,
                                    top_k=-1 if generation_config.top_k == 0 else generation_config.top_k,
                                    temperature=generation_config.temperature,
                                    repetition_penalty=generation_config.repetition_penalty,
                                    max_tokens=generation_config.max_new_tokens)
    # vLLM异步推理（在独立线程中阻塞执行推理，主线程异步等待完成通知）
    results_iter=engine.generate(prompt=None,sampling_params=sampling_params,prompt_token_ids=prompt_tokens,request_id=str(uuid.uuid4().hex))
    
    # 流式返回，即迭代transformer的每一步推理结果并反复返回
    if stream:
        async def streaming_resp():
            async for result in results_iter:
                text=result.outputs[0].text
                for token_id in stop_words_ids:
                    text=text.replace(tokenizer.decode(token_id),'') 
                yield (json.dumps({'text':text})+'\0').encode('utf-8')
        return StreamingResponse(streaming_resp())

    # 整体返回
    async for result in results_iter:
        text=result.outputs[0].text
    
    # 清理停止词
    for token_id in stop_words_ids:
        text=text.replace(tokenizer.decode(token_id),'')
    ret={"text":text}
    return JSONResponse(ret)

if __name__=='__main__':
    uvicorn.run(app,
                host=None,
                port=8000,
                log_level="debug")