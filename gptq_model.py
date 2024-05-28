import os


import json
import argparse
import logging
import random
from datasets import Dataset
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from transformers import AutoTokenizer
import time
device = "cuda" # the device to load the model onto
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"
max_len = 8192
# CUDA_VISIBLE_DEVICES=0 python gptq_model.py
import torch
def load_model_tokenizer(model_path): 
    quantize_config = BaseQuantizeConfig(
        bits=3, # 4 or 8
        group_size=128,
        damp_percent=0.1,
        desc_act=False,  # set to False can significantly speed up inference but the perplexity may slightly bad
        static_groups=False,
        sym=True,
        true_sequential=True,
        model_name_or_path=None,
        model_file_base_name="model"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)
    model = AutoGPTQForCausalLM.from_pretrained(model_path, quantize_config=quantize_config,trust_remote_code=True,device_map="auto",)

    return model, tokenizer


def get_gptq_data(data_path, tokenizer, n_samples):
    with open(data_path, 'r', encoding='utf-8') as file:
        messages = json.load(file)

    data = []
    for msg in messages:
        # msg = c['messages']
        # print('msg', msg['messages'])
        text = tokenizer.apply_chat_template(msg['messages'], tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text])
        input_ids = torch.tensor(model_inputs.input_ids[:max_len], dtype=torch.int)
        # attention_mask = torch.tensor(model_inputs.attention_mask[:max_len], dtype=torch.int)
        # attention_mask=input_ids.ne(tokenizer.pad_token_id)
        # print('model_inputs', model_inputs)
        # print('attention_mask', attention_mask)
        # print('===========================')
        data.append(dict(input_ids=input_ids, attention_mask=input_ids.ne(tokenizer.pad_token_id)))
        # data.append(dict(input_ids=input_ids, attention_mask=attention_mask))
    # 创建了一个与 input_ids 长度相同的列表,其中对应位置的元素如果不等于 tokenizer.pad_token_id,则为 True,否则为 False。


    data = random.sample(data, k=min(n_samples, len(data))) #随机选样本

    return data


def load_data(data_path, tokenizer, n_samples):
    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    raw_data = random.sample(raw_data, k=min(n_samples, len(raw_data)))

    def dummy_gen():
        return raw_data

    def tokenize(examples):
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]

        prompts = []
        texts = []
        input_ids = []
        attention_mask = []
        for istr, inp, opt in zip(instructions, inputs, outputs):
            if inp:
                prompt = f"Instruction:\n{istr}\nInput:\n{inp}\nOutput:\n"
                text = prompt + opt
            else:
                prompt = f"Instruction:\n{istr}\nOutput:\n"
                text = prompt + opt
            if len(tokenizer(prompt)["input_ids"]) >= tokenizer.model_max_length:
                continue

            tokenized_data = tokenizer(text)

            input_ids.append(tokenized_data["input_ids"][: tokenizer.model_max_length])
            attention_mask.append(tokenized_data["attention_mask"][: tokenizer.model_max_length])
            prompts.append(prompt)
            texts.append(text)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "prompt": prompts,
        }

    dataset = Dataset.from_generator(dummy_gen)

    dataset = dataset.map(
        tokenize,
        batched=True,
        batch_size=len(dataset),
        num_proc=1,
        keep_in_memory=True,
        load_from_cache_file=False,
        remove_columns=["instruction", "input"],
    )

    dataset = dataset.to_list()

    for sample in dataset:
        sample["input_ids"] = torch.LongTensor(sample["input_ids"])
        sample["attention_mask"] = torch.LongTensor(sample["attention_mask"])
    
    examples_for_quant = [
        {"input_ids": example["input_ids"], "attention_mask": example["attention_mask"]} for example in dataset
    ]

    return examples_for_quant


def main(args):
    model_path = args.model_path
    quant_path = args.quant_path
    data_path = args.data_path
    n_samples = args.num_samples

    model, tokenizer = load_model_tokenizer(model_path)

    data = get_gptq_data(data_path, tokenizer, n_samples)
    # data = load_data("../../gptq/dataset/alpaca_data_cleaned.json", tokenizer, n_samples)

    logging.basicConfig(
       format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
    )
    model.quantize(data, cache_examples_on_gpu=False)

    model.save_quantized(quant_path, use_safetensors=True)
    tokenizer.save_pretrained(quant_path)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model = AutoGPTQForCausalLM.from_quantized(
        quant_path,
        # device="cuda:4",
        # device_map={"": device},
        device_map = "auto", 
        use_triton=False,
        # max_memory=max_memory,
        inject_fused_mlp=True,
        inject_fused_attention=True,
        trust_remote_code=True,
    )
    
    
    start = time.time()
    
    print(tokenizer.decode(model.generate(**tokenizer("星巴克是一个?", return_tensors="pt").to(model.device),max_length=600)[0]))

    end = time.time()

    print("量化后模型推理时间消耗：", end - start, "秒")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test HF checkpoint.")
    parser.add_argument(
        "-m",
        "--model_path",
        type=str,
        default="/data0/zhangchongrui/model/compused",
    )
    parser.add_argument(
        "-q",
        "--quant_path",
        type=str,
        default="/data0/zhangchongrui/model/GPTQ_14B_3bit_128g",
    )
    parser.add_argument(
        "-d",
        "--data_path",
        type=str,
        default="/data0/zhangchongrui/dataset/RAG_instruction_105_Q.json",
    )
    parser.add_argument("-s", "--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--max_len", type=int, default=8192, help="max_len")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=128,
        help="how many samples will be used to quantize model",
    )

    args = parser.parse_args()

    main(args)
