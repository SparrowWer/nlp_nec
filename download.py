import torch
from modelscope import snapshot_download, AutoModel, AutoTokenizer
import os
# model_dir = snapshot_download('deepseek-ai/DeepSeek-V2-Chat', cache_dir='/data0/zhangchongrui/model/DeepSeek-V2-Chat', revision='master')

#python -m vllm.entrypoints.openai.api_server --model /data0/zhangchongrui/model/Qwen-72B-Chat-Int4 --served-model-name Qwen-72B-Chat-Int4 --max-model-len=2048 --trust-remote-code
# curl http://localhost:8000/v1/completions \
#     -H "Content-Type: application/json" \
#     -d '{
#         "model": "Qwen-72B-Chat-Int4",
#         "prompt": "有没有推荐的自然风光",
#         "max_tokens": 256,
#         "temperature": 1
#     }'
from openai import OpenAI
openai_api_key = "EMPTY" # 随便设，只是为了通过接口参数校验
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

chat_outputs = client.chat.completions.create(
    model="Qwen-72B-Chat-Int4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "你好"},
    ]
)
print(chat_outputs)