import gradio as gr
import requests
import json

MAX_HISTORY_LEN=50

def chat_streaming(query,history):
    # 调用api_server
    response=requests.post('http://localhost:8000/chat',json={
        'query':query,
        'stream': True,
        'history':history
    },stream=True)
    
    # 流式读取http response body, 按\0分割
    for chunk in response.iter_lines(chunk_size=8192,decode_unicode=False,delimiter=b"\0"):
        if chunk:
            data=json.loads(chunk.decode('utf-8'))
            text=data["text"].rstrip('\r\n') # 确保末尾无换行
            yield text

with gr.Blocks(css='.qwen-logo img {height:200px; width:600px; margin:0 auto;}') as app:
    with gr.Row():
        logo_img=gr.Image('qwen.png',elem_classes='qwen-logo')
    with gr.Row():
        chatbot=gr.Chatbot(label='通义千问14B-Chat-Int4')
    with gr.Row():
        query_box=gr.Textbox(label='提问',autofocus=True,lines=5)
    with gr.Row():
        clear_btn=gr.ClearButton([query_box,chatbot],value='清空历史')
        submit_btn=gr.Button(value='提交')

    def chat(query,history):
        for response in chat_streaming(query,history):
            yield '',history+[(query,response)]
        history.append((query,response))
        while len(history)>MAX_HISTORY_LEN:
            history.pop(0)
    
    # 提交query
    submit_btn.click(chat,[query_box,chatbot],[query_box,chatbot])
    # query_box.submit(chat,[query_box,chatbot],[query_box,chatbot])

if __name__ == "__main__":
    app.queue(200)  # 请求队列
    app.launch(server_name='0.0.0.0',max_threads=500) # 线程池