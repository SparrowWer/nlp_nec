from vllm import LLM, SamplingParams
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from auto_gptq import AutoGPTQForCausalLM
import torch
# Sample prompts.
import os
name = "/data0/zhangchongrui/model/GPTQ_14B_4bit_128g"
gen_num=256
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
def load_models_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(
        name,
        trust_remote_code=True
    )
    # model = AutoModelForCausalLM.from_pretrained(
    # model_name_or_path,
    # device_map="auto",
    # )
    model = AutoGPTQForCausalLM.from_quantized(
    name,
    device_map = "auto", 
    use_triton=False,
    inject_fused_mlp=True,
    inject_fused_attention=True,
    trust_remote_code=True,
    )
    model.eval()
    model.generation_config = GenerationConfig.from_pretrained(
    name,
    trust_remote_code=True
    )
    return model, tokenizer
# Create a sampling params object.

sampling_params = SamplingParams(temperature=0.8, top_p=0.95,max_tokens=gen_num) #设置文本生成的采样参数。它遵循OpenAI文本完成API的采样参数，并增加了对beam search（束搜索）的支持，而这在OpenAI中是不支持的。
llm = LLM(model=name,trust_remote_code=True, tensor_parallel_size=2,gpu_memory_utilization=0.8,max_model_len=2048)
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)

start = time.perf_counter()
total_tokens = 0
max_gpu_memory_cost=0
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    total_tokens += len(generated_text.split())  # count the number of tokens in the generated text
    max_gpu_memory_cost = max(max_gpu_memory_cost, torch.cuda.max_memory_allocated())
    torch.cuda.empty_cache()
end = time.perf_counter()

print(f"GPU Memory cost: {max_gpu_memory_cost / 1024 / 1024 / 1024}GB")
print("运行时间：", end-start, "秒")
print("Average generate speed (tokens/s): {}".format(total_tokens / (end-start)))


# 不使用vllm
# time_costs=[]
# model, tokenizer = load_models_tokenizer()
# config = GenerationConfig.from_pretrained(name, trust_remote_code=True)
# config.max_new_tokens = gen_num
# for context_str in prompts:
#     inputs = tokenizer(context_str, return_tensors='pt')
#     inputs = inputs.to(model.device)
#     t1 = time.time()
#     pred = model.generate(**inputs, generation_config=config)
#     time_costs.append(time.time() - t1)
#     max_gpu_memory_cost = max(0, torch.cuda.max_memory_allocated())
#     torch.cuda.empty_cache()
# print("Average generate speed (tokens/s): {}".format((1 * gen_num * len(prompts)) / sum(time_costs)))
# print(f"GPU Memory cost: {max_gpu_memory_cost / 1024 / 1024 / 1024}GB")
