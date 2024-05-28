
# 测评RM以及推理速度
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer,BitsAndBytesConfig
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from transformers.generation import GenerationConfig
from transformers.trainer_utils import set_seed
from tqdm import tqdm
from peft import PeftModel
seed = 1024
max_experiment_times = 1
context_length_per_experiment = 1
generate_length_per_experiment = 2048
#运行
# CUDA_VISIBLE_DEVICES="0,1" python profile_test.py
use_flash_attn = True
# model_name_or_path = "/data0/zhangchongrui/model/Qwen1.5_72B_models/Qwen1.5-72B-Chat"
# adapter_path = "/data0/zhangchongrui/model/qlora_bnb_adapter"
#lora合并后模型
model_name_or_path ="/data0/zhangchongrui/model/GPTQ_72B_3bit_128g"
#普通模型加载示例
def load_models_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        # pad_token='<|extra_0|>',
        # eos_token='<|endoftext|>',
        # bos_token='<|im_start|>',
        # padding_side='left',
        trust_remote_code=True
    )
    # model = AutoModelForCausalLM.from_pretrained(
    # model_name_or_path,
    # device_map="auto",
    # )
    model = AutoGPTQForCausalLM.from_quantized(
    model_name_or_path,
    # device="cuda:4",
    # device_map={"": device},
    device_map = "auto", 
    use_triton=False,
    # max_memory=max_memory,
    inject_fused_mlp=True,
    inject_fused_attention=True,
    trust_remote_code=True,
    )
    model.eval()
    model.generation_config = GenerationConfig.from_pretrained(
    model_name_or_path,
    pad_token_id=tokenizer.pad_token_id,
    trust_remote_code=True
    )

    return model, tokenizer
#qlora方法（bnb）加载模型示例
# def load_models_tokenizer():
#     tokenizer = AutoTokenizer.from_pretrained(
#         model_name_or_path,
#         pad_token='<|extra_0|>',
#         eos_token='<|endoftext|>',
#         bos_token='<|im_start|>',
#         padding_side='left',
#         trust_remote_code=True
#     )
#     model = AutoModelForCausalLM.from_pretrained(
#     model_name_or_path,
#     torch_dtype=torch.bfloat16,
#     device_map="auto",
#     quantization_config=BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_compute_dtype=torch.bfloat16,
#         bnb_4bit_use_double_quant=True,
#         bnb_4bit_quant_type='nf4',
#     )
#     )
#     model = PeftModel.from_pretrained(model, adapter_path)
#     model.eval()
#     model.generation_config = GenerationConfig.from_pretrained(
#     model_name_or_path,
#     pad_token_id=tokenizer.pad_token_id,
#     trust_remote_code=True
#     )

#     return model, tokenizer

#lora方法加载模型示例
# def load_models_tokenizer():
    # tokenizer = AutoTokenizer.from_pretrained(
    #     "/data0/zhangchongrui/model/Qwen1.5_72B_models/Qwen1.5-72B-Chat",
    #     pad_token='<|extra_0|>',
    #     eos_token='<|endoftext|>',
    #     bos_token='<|im_start|>',
    #     padding_side='left',
    #     trust_remote_code=True
    # )
    # model = AutoModelForCausalLM.from_pretrained(
    # model_name_or_path,
    # torch_dtype=torch.bfloat16,
    # device_map={"": 0},
    # )
    # model = PeftModel.from_pretrained(model, adapter_path)
    # model.eval()
#     model = AutoGPTQForCausalLM.from_quantized(
#     model_name_or_path,
#     # device="cuda:4",
#     # device_map={"": device},
#     device_map = "auto", 
#     use_triton=False,
#     # max_memory=max_memory,
#     inject_fused_mlp=True,
#     inject_fused_attention=True,
#     trust_remote_code=True,
# )
    # model.generation_config = GenerationConfig.from_pretrained(
    #     model_name_or_path,
    #     pad_token_id=tokenizer.pad_token_id,
    #     trust_remote_code=True
    # )

    # return model, tokenizer
set_seed(seed)

model, tokenizer = load_models_tokenizer()

# Specify hyperparameters for generation
config = GenerationConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
config.min_length = generate_length_per_experiment + context_length_per_experiment
config.max_new_tokens = generate_length_per_experiment

time_costs = []
context_str = '我' * context_length_per_experiment
max_gpu_memory_cost = 0
for _ in tqdm(range(max_experiment_times)):
    inputs = tokenizer(context_str, return_tensors='pt')
    inputs = inputs.to(model.device)
    t1 = time.time()
    pred = model.generate(**inputs, generation_config=config)
    time_costs.append(time.time() - t1)
    assert pred.shape[1] == config.min_length
    max_gpu_memory_cost = max(max_gpu_memory_cost, torch.cuda.max_memory_allocated())
    torch.cuda.empty_cache()

print("Average generate speed (tokens/s): {}".format((max_experiment_times * generate_length_per_experiment) / sum(time_costs)))
print(f"GPU Memory cost: {max_gpu_memory_cost / 1024 / 1024 / 1024}GB")
print("Experiment setting: ")
print(f"seed = {seed}")
print(f"max_experiment_times = {max_experiment_times}")
print(f"context_length_per_experiment = {context_length_per_experiment}")
print(f"generate_length_per_experiment = {generate_length_per_experiment}")
print(f"use_flash_attn = {use_flash_attn}")
