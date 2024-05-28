import os
 
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"
 
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
 
"""
将lora权重合并到大模型中
"""
#   CUDA_VISIBLE_DEVICES=1 python lora_compose.py
def merge_lora_to_LLM():
    model_name_or_path = "/data0/da/Qwen1.5_14B_models/Qwen1.5-14B-Chat"
    adapter_name_or_path = "/data0/da/Qwen1.5_14B_models/peft_qwen14b_rag105_ckpt12"
    save_path = "/data0/zhangchongrui/model/compused_14B"
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(model, adapter_name_or_path)
    model = model.merge_and_unload()
 
    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)
 
if __name__ == "__main__":
    merge_lora_to_LLM()