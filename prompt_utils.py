import copy

def tokenize_and_encode(tokenizer, role, content, im_start_id, im_end_id, nl_token_id):
    """
    为给定的角色和内容编码并包装，返回编码后的文本和token ids。
    """
    role_encoded = tokenizer.encode(f"{role}")
    content_encoded = tokenizer.encode(content)
    wrapped_tokens = [im_start_id] + role_encoded + [nl_token_id] + content_encoded + [im_end_id] + [nl_token_id]
    wrapped_text = f"{role}\n{content}\n"
    return wrapped_text, wrapped_tokens

def _build_prompt(generation_config, tokenizer, query, history=None, system=""):
    if history is None:
        history = []
    
    nl_token_id = tokenizer.encode("\n")[0]
    im_start_id = tokenizer.im_start_id
    im_end_id = tokenizer.im_end_id
    left_token_space = generation_config.max_window_size
    
    # 处理系统声明
    # <im_start>system\nYou are a helpful assistant<im_end>\n
    system_text, system_tokens = tokenize_and_encode(tokenizer, "system", system, im_start_id, im_end_id, nl_token_id)
    left_token_space -= len(system_tokens)
    
    # 处理用户查询部分
    # <im_start>system\nWho you are<im_end>\n
    query_text, query_tokens = tokenize_and_encode(tokenizer, "user", query, im_start_id, im_end_id, nl_token_id)
    # <im_start>assistant\n
    assistant_tokens_prefix = [im_start_id] + tokenizer.encode("assistant") + [nl_token_id]
    
    # 保证用户查询可以完整地包含在内
    if len(query_tokens) + len(assistant_tokens_prefix) > left_token_space:
        query_token_len = left_token_space - len(assistant_tokens_prefix)
        query_tokens = query_tokens[:query_token_len]
        query_text = tokenizer.decode(query_tokens)
    # <im_start>system\nWho you are<im_end>\n<im_start>assistant\n
    query_tokens += assistant_tokens_prefix
    query_text += "assistant\n"
    left_token_space -= len(query_tokens)

    # 处理历史对话部分
    history_text, history_tokens = '', []
    for hist_query, hist_response in reversed(history):
        if left_token_space <= 0:
            break
        hist_query_text, hist_query_tokens = tokenize_and_encode(tokenizer, "user", hist_query, im_start_id, im_end_id, nl_token_id)
        hist_response_text, hist_response_tokens = tokenize_and_encode(tokenizer, "assistant", hist_response, im_start_id, im_end_id, nl_token_id)
        # 当前所有的历史
        cur_hist_tokens = hist_query_tokens + hist_response_tokens
        if len(cur_hist_tokens) <= left_token_space:
            history_text = f"{hist_query_text}\n{hist_response_text}\n" + history_text
            history_tokens = cur_hist_tokens + history_tokens
            left_token_space -= len(cur_hist_tokens)
        else:
            break
            
    prompt_str = f'{system_text}{history_text}{query_text}'
    prompt_tokens = system_tokens + history_tokens + query_tokens
    return prompt_str, prompt_tokens

def remove_stop_words(token_ids, stop_words_ids):
    return [token for token in token_ids if token not in stop_words_ids]