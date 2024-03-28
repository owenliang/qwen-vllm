# 按chatml格式构造千问的Prompt
def _build_prompt(
        generation_config,
        tokenizer,
        query,
        history=None,
        system=""):
    if history is None:
        history = []

    # 包裹发言内容的token
    im_start, im_start_tokens = '<|im_start|>', [151644]
    im_end, im_end_tokens = '<|im_end|>', [151645]

    # 换行符token
    nl_tokens = tokenizer.encode("\n")

    # 用于编码system/user/assistant的一段发言, 格式{role}\n{content}
    def _tokenize_str(role, content):  # 返回元组，下标0是文本，下标1是token ids
        return f"{role}\n{content}", tokenizer.encode(role) + nl_tokens + tokenizer.encode(content)

    # 剩余token数
    left_token_space = generation_config.max_window_size

    # prompt头部: system发言
    system_text_part, system_tokens_part = _tokenize_str("system",
                                                         system)  # system_tokens_part -->    system\nYou are a helpful assistant.
    system_text = f'{im_start}{system_text_part}{im_end}'
    system_tokens = im_start_tokens + system_tokens_part + im_end_tokens  # <|im_start|>system\nYou are a helpful assistant.<|im_end|>
    left_token_space -= len(system_tokens)

    # prompt尾部: user发言和assistant引导
    query_text_part, query_tokens_part = _tokenize_str('user', query)
    query_tokens_prefix = nl_tokens + im_start_tokens
    query_tokens_suffix = im_end_tokens + nl_tokens + im_start_tokens + tokenizer.encode('assistant') + nl_tokens
    if len(query_tokens_prefix) + len(query_tokens_part) + len(query_tokens_suffix) > left_token_space:  # query太长截断
        query_token_len = left_token_space - len(query_tokens_prefix) - len(query_tokens_suffix)
        query_tokens_part = query_tokens_part[:query_token_len]
        query_text_part = tokenizer.decode(query_tokens_part)
    query_tokens = query_tokens_prefix + query_tokens_part + query_tokens_suffix
    query_text = f"\n{im_start}{query_text_part}{im_end}\n{im_start}assistant\n"
    left_token_space -= len(query_tokens)

    # prompt腰部: 历史user+assitant对话
    history_text, history_tokens = '', []
    for hist_query, hist_response in reversed(history):  # 优先采用最近的对话历史
        hist_query_text, hist_query_tokens_part = _tokenize_str("user", hist_query)  # user\n历史提问
        hist_response_text, hist_response_tokens_part = _tokenize_str("assistant", hist_response)  # assistant\n历史回答
        # 生成本轮对话
        cur_history_tokens = nl_tokens + im_start_tokens + hist_query_tokens_part + im_end_tokens + nl_tokens + im_start_tokens + hist_response_tokens_part + im_end_tokens
        cur_history_text = f"\n{im_start}{hist_query_text}{im_end}\n{im_start}{hist_response_text}{im_end}"
        # 储存多轮对话
        if len(cur_history_tokens) <= left_token_space:
            history_text = cur_history_text + history_text
            history_tokens = cur_history_tokens + history_tokens
        else:
            break

            # 生成完整Prompt
    prompt_str = f'{system_text}{history_text}{query_text}'
    prompt_tokens = system_tokens + history_tokens + query_tokens
    return prompt_str, prompt_tokens
