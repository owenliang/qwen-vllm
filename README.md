# qwen-vllm

[千问官方部署文档](https://github.com/QwenLM/Qwen?tab=readme-ov-file#deployment)

vllm_wrapper.py实现参考了[Qwen官方实现](https://github.com/QwenLM/Qwen/blob/main/examples/vllm_wrapper.py)

## 安装注意

- python版本为3.10
- cuda版本是12.1
- torch安装2.1
- 安装vllm gptq量化版, 安装时命令采用pip install . -i https://mirrors.aliyun.com/pypi/simple/
- 安装modelscope，命令pip install modelscope -i https://mirrors.aliyun.com/pypi/simple/
- 安装千问的toktoken分词库 pip install tiktoken -i https://mirrors.aliyun.com/pypi/simple/