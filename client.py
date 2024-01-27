from vllm_wrapper import vLLMWrapper

model = "qwen/Qwen-7B-Chat-Int4"

vllm_model = vLLMWrapper(model,
                            quantization = 'gptq',
                            dtype="float16",
                            tensor_parallel_size=1)

history=None 
while True:
    Q=input('提问:')
    response, history = vllm_model.chat(query=Q,
                                        history=history)
    print(response)
    history=history[:5]