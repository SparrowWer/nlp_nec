import os,time

from transformers import AutoModelForCausalLM, AutoTokenizer,BitsAndBytesConfig
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from transformers.generation import GenerationConfig
from transformers.trainer_utils import set_seed
from tqdm import tqdm
from peft import PeftModel
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
# CUDA_VISIBLE_DEVICES="4,5" python test.py
model_name_or_path ="/data0/zhangchongrui/model/GPTQ_14B_4bit_128g"
# context_str="您将扮演一个严谨且忠实于所提供的背景信息的客服助手。您的任务是根据提供的背景信息回答用户的问题。请仔细阅读并记住下面的背景信息，然后根据这些背景信息，用中文提供准确的结论性回答，但是如果问题的答案不包含在背景信息中，请回答“根据已知信息无法回答该问题，请联系人工客服。”。在回答之前，请使用“思考链条”来组织您的思考过程，但只需提供思考链条的结论部分。请确保回答的准确性。\n\n背景信息：\n问题：SDMS系统中，资产团队的FieldAssetsSpecialist或FieldAssetsManager角色需要进行什么样的团队维护？答案：在SDMS系统中，资产团队的FieldAssetsSpecialist或FieldAssetsManager角色需要进行团队上下级关系的维护。\n\n问题：SDMS系统中，哪些用户角色需要进行团队上下级维护？答案：SDMS系统中需要进行团队上下级维护的用户角色包括：设计团队的所有角色，开发中的DealMaker，以及资产团队的FieldAssetsSpecialist或FieldAssetsManager。\n\n回答步骤：\n1. 仔细阅读并保证已经充分理解提供的背景信息。\n2. 使用“思考链条”根据提供的背景信息逐步分析问题，但在回答时仅提供结论。\n3. 如果所需信息在提供的背景信息中，请用中文提供准确的结论性回答。如果所需信息不在提供的背景信息中，请明确指出“根据已知信息无法回答该问题，请联系人工客服。”。\n4. 确保回答的准确性。\n\n现在，请根据上述步骤回答用户的问题，要回答的问题是：SDMS系统中，资产团队的FieldAssetsSpecialist或FieldAssetsManager角色需要进行什么样的团队维护？ 在SDMS系统中，资产团队的FieldAssetsSpecialist或FieldAssetsManager角色需要进行团队上下级关系的维护。"
context_str= "Here are the main ideas of Jeff Walker's Product Launch Formula that can be applied by a growth marketing agency for their clients:\n\n1. Identify the target audience and their needs: Understand the ideal customer for the product or service, and create a messaging that resonates with them.\n2. Pre-launch: Build anticipation and excitement for the launch by creating buzz, gathering testimonials and case studies, and using social media to create awareness.\n3. Launch: Use a well-crafted launch sequence to maximize sales and conversions. This can include offering bonuses, creating scarcity, and using a deadline to create urgency.\n4. Post-launch: Follow up with customers, gather feedback, and continue to provide value to keep them engaged and loyal.\n5. Create a product suite: Once the initial product is successful, expand the offering by creating additional products or services that address other needs of the same audience.\n6. Continual optimization: Continually monitor and optimize the launch process and product suite to improve results.\n7. Build a community: Use the launch process to build a community of customers who are passionate about the product and can help spread the word.\n8. Use automation: Use technology and automation to streamline the launch process and improve efficiency."
def load_models_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        pad_token='<|extra_0|>',
        eos_token='<|endoftext|>',
        bos_token='<|im_start|>',
        padding_side='left',
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
    model.generation_config = GenerationConfig.from_pretrained(#"max_new_tokens": 512
    model_name_or_path,
    trust_remote_code=True
    )

    return model, tokenizer

generate_length_per_experiment = 1
context_length_per_experiment =1024
config = GenerationConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
config.max_new_tokens = generate_length_per_experiment +context_length_per_experiment
model, tokenizer = load_models_tokenizer()
inputs = tokenizer(context_str, return_tensors='pt')
inputs = inputs.to(model.device)  # Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!  之前model在cuda:0上，而inputs在cpu上
# print(inputs)
print(len(inputs[0])) #380
# print(tokenizer.decode([ 87026,  44063, 102889,  46944,108487, 100136, 114538,  34204, 113460])) #您将扮演一个严谨且忠实于所提供的
pred = model.generate(**inputs,generation_config=config)
print(tokenizer.decode(pred[0]))