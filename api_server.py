import os 
from vllm import AsyncEngineArgs,AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from modelscope import AutoTokenizer, GenerationConfig
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
import uvicorn
from prompt_tpl import _build_prompt
import uuid 

# http接口服务
app=FastAPI()

# vLLM参数
model_dir="qwen/Qwen-7B-Chat-Int4"
tensor_parallel_size=1
gpu_memory_utilization=0.6
quantization='gptq'
dtype='float16'

# vLLM加载模型
generation_config = GenerationConfig.from_pretrained(model_dir,trust_remote_code=True)
# 加载分词器
tokenizer=AutoTokenizer.from_pretrained(model_dir,trust_remote_code=True)
tokenizer.eos_token_id=generation_config.eos_token_id
# 推理终止词，遇到这些词停止继续推理
stop_words_ids=[tokenizer.im_start_id,tokenizer.im_end_id,tokenizer.eos_token_id]
# vLLM参数
args=AsyncEngineArgs(model_dir)
args.worker_use_ray=False
args.engine_use_ray=False
args.tokenizer=model_dir
args.tensor_parallel_size=1
args.trust_remote_code=True
args.quantization=quantization
args.gpu_memory_utilization=gpu_memory_utilization
args.dtype=dtype
# vLLM加载模型
os.environ['VLLM_USE_MODELSCOPE']='True'
engine=AsyncLLMEngine.from_engine_args(args)


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
    async for request_output in results_iter:
        final_output=request_output

    # 清除可能出现的内置终止词
    response=final_output.outputs[0].text
    for token_id in stop_words_ids:
        response=response.replace(tokenizer.decode(token_id),'')
    ret = {"text":response}
    return JSONResponse(ret)

if __name__=='__main__':
    uvicorn.run(app,
                host=None,
                port=8000,
                log_level="debug")