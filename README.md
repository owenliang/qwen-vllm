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

## 底层原理

1.8B预训练版本，训练数据：

- 语料：[百度文库](https://wenku.baidu.com/view/11188178.html)

输入：英国航空，中文简称英航，是英国的国家航空公司，也是寰宇一家的创始成员及国际航空集团旗下子公司。<|endoftext|>
输出：英航的主要枢纽为伦敦希思罗机场及伦敦盖特威克机场。英航是欧洲第二大的航空公司、西欧最大的航空公司及全球三间其中一间曾拥有协和客机的航空公司，其余两间为法国航空和新加坡航空。<|endoftext|>

1.8B-Chat版本，基于1.8B预训练版本进行微调（SFT，S监督学习，FT微调）训练数据：

输入：<|im_start|>system\nyou are ahelper assitant.\n<|im_end|>
\n<|im_start|>user\n历史提问A？\n<|im_end|><|im_start|>assitant:历史回答A\n<|im_end|>
\n<|im_start|>user\n历史提问B？\n<|im_end|><|im_start|>assitant:历史回答B\n<|im_end|>
\n<|im_start|>user\n了解英国航空么？\n<|im_end|><|im_start|>assitant:\n<|endoftext|>
输出：英国航空，中文简称英航，是英国的国家航空公司。<|im_end|><|endoftext|>
