from transformers import AutoModelForCausalLM, AutoTokenizer
# from peft import PeftModel
import psutil
import torch
# CUDA_VISIBLE_DEVICES="0,5" python qwenchat.py
def convert_bytes_to_gb(bytes_amount):
    """将字节转换为GB"""
    gb = bytes_amount / (1024.0 ** 3)
    return round(gb, 2)


device = "cuda" # 加载模型的设备
model_name_or_path="/data0/zhangchongrui/model/Qwen-14B-Chat"
# model_name_or_path = '/data0/zhangchongrui/model/Qwen1.5_72B_models/Qwen1.5-72B-Chat'
# adapter_path = '/data0/zhangchongrui/model/Qwen1.5_72B_models/peft_qwen72b_rag105_ckpt6'
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    device_map="auto",
    trust_remote_code=True
)
# model = PeftModel.from_pretrained(model, adapter_path)
model_size_gb = convert_bytes_to_gb(torch.cuda.memory_allocated())
print(f"Model loaded on GPU, size: {model_size_gb} GB")

model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

# prompt = "您将扮演一个严谨且忠实于所提供的背景信息的客服助手。您的任务是根据提供的背景信息回答用户的问题。请仔细阅读并记住下面的背景信息，然后根据这些背景信息，\
#     用中文提供准确的结论性回答，但是如果问题的答案不包含在背景信息中，请回答“根据已知信息无法回答该问题，请联系人工客服。”。\
#     在回答之前，请使用“思考链条”来组织您的思考过程，但只需提供思考链条的结论部分。请确保回答的准确性。\n\n\
#         背景信息：问题：如何解决Citrix系统登录密码忘记的问题？答案：如果您忘记了Citrix系统的登录密码，可以尝试以下方法：（1）如果您在谷歌浏览器中保存了密码，可以按照以下步骤找回：a.打开Citrix图标，跳转到谷歌页面后，点击页面右上角的4个小点。b.打开后选择设置。c.进入后选择密码。d.从弹出的选项中，选中与门店Citrix账号相同的账号，点击小眼睛图标。e.输入门店的开机密码Welcome1进行查看即可。（2）如果您没有在谷歌浏览器中保存密码，请联系IT部门电话400-628-7289。如果登录时遇到其他问题，如弹出代理服务器登录页面，建议您注销登录状态，清空浏览器缓存，切换服务器。如果问题仍然存在，请联系IT部门电话400-628-7289。\n\n\
#             问题：Citrix密码忘记，答案：如果您忘记了Citrix系统的登录密码，可以尝试以下方法：（1）如果您在谷歌浏览器中保存了密码，可以按照以下步骤找回：a.打开Citrix图标，跳转到谷歌页面后，点击页面右上角的4个小点。b.打开后选择设置。c.进入后选择密码。d.从弹出的选项中，选中与门店Citrix账号相同的账号，点击小眼睛图标。e.输入门店的开机密码Welcome1进行查看即可。（2）如果您没有在谷歌浏览器中保存密码，请联系IT\n\n\
#                 回答步骤：\n1. 仔细阅读并保证已经充分理解提供的背景信息。\n2. 使用“思考链条”根据提供的背景信息逐步分析问题，但在回答时仅提供结论。\n3. 如果所需信息在提供的背景信息中，请用中文提供准确的结论性回答。如果所需信息不在提供的背景信息中，请明确指出“根据已知信息无法回答该问题，请联系人工客服。”。\n4. 确保回答的准确性。\n\n\
#                     现在，请根据上述步骤回答用户的问题，要回答的问题是：Citrix系统登录时如何输入账户密码？"

# prompt = "您将扮演一个严谨且忠实于所提供的背景信息的客服助手。您的任务是根据提供的背景信息回答用户的问题。请仔细阅读并记住下面的背景信息，然后根据这些背景信息，\
#     用中文提供准确的结论性回答，但是如果问题的答案不包含在背景信息中，请回答“根据已知信息无法回答该问题，请联系人工客服。”。\
#     在回答之前，请使用“思考链条”来组织您的思考过程，但只需提供思考链条的结论部分。请确保回答的准确性。\n\n\
#         背景信息：问题：汇集小票机不出纸，答案：处理方法如下：a、请门店清理汇集打印机纸屑，操作包括打开机盖、拆下纸卷、按压住前挡片、倒转并轻轻拍打完成清理碎纸片。B、重启有问题的汇集小票机所连接的POS机。如果问题仍然存在，请联系人工客服，IT部门电话400-628-7289。\n\n\
#             问题：汇集小票机不出票，应该如何处理？答案：处理方法如下：a、请门店清理汇集打印机纸屑，操作包括打开机盖、拆下纸卷、按压住前挡片、倒转并轻轻拍打完成清理碎纸片。B、重启有问题的汇集小票机所连接的POS机。如果问题仍然存在，请联系人工客服，IT部门电话400-628-7289。\n\n\
#                 回答步骤：\n1. 仔细阅读并保证已经充分理解提供的背景信息。\n2. 使用“思考链条”根据提供的背景信息逐步分析问题，但在回答时仅提供结论。\n3. 如果所需信息在提供的背景信息中，请用中文提供准确的结论性回答。如果所需信息不在提供的背景信息中，请明确指出“根据已知信息无法回答该问题，请联系人工客服。”。\n4. 确保回答的准确性。\n\n\
#                     现在，请根据上述步骤回答用户的问题，要回答的问题是：汇集小票打印机为何自动不出纸，手动却能正常？"

# prompt = "您将扮演一个严谨且忠实于所提供的背景信息的客服助手。您的任务是根据提供的背景信息回答用户的问题。请仔细阅读并记住下面的背景信息，然后根据这些背景信息，\
#     用中文提供准确的结论性回答，但是如果问题的答案不包含在背景信息中，请回答“根据已知信息无法回答该问题，请联系人工客服。”。\
#     在回答之前，请使用“思考链条”来组织您的思考过程，但只需提供思考链条的结论部分。请确保回答的准确性。\n\n\
#         背景信息：问题：如何在Citrix系统中修改即将过期的密码？答案：当你尝试登入Citrix时，网站会转跳到一个提示更改密码的页面。在此页面，你需要输入用户名和新旧密码，然后点击[确定]。新密码需要注意以下几点：\
# 1.需要有英文大小写\
# 2.尽量不要使用标点符号，因涉及半角全角，故不建议使用标点符号\
# 3.需要满足最少8位数\
# 4.不要使用之前使用过的密码\
# 完成后，你可以正常使用CitrixAPP。下次访问Citrix时，请使用新密码登录。\n\n\
#             问题：Citrix密码修改，答案：当您修改密码时，需要注意以下几点\
# 1.需要有英文大小写\
# 2.尽量不要使用标点符号，因涉及半角全角，故不建议使用标点符号\
# 3.需要满足最少8位数\
# 4.不要使用之前使用过的密码\
# 请您依照上述条件进行修改，谢谢。\
#                 回答步骤：\n1. 仔细阅读并保证已经充分理解提供的背景信息。\n2. 使用“思考链条”根据提供的背景信息逐步分析问题，但在回答时仅提供结论。\n3. 如果所需信息在提供的背景信息中，请用中文提供准确的结论性回答。如果所需信息不在提供的背景信息中，请明确指出“根据已知信息无法回答该问题，请联系人工客服。”。\n4. 确保回答的准确性。\n\n\
#                     现在，请根据上述步骤回答用户的问题，要回答的问题是：如何更新Citrix系统的密码？"
# prompt = "您将扮演一个严谨且忠实于所提供的背景信息的客服助手。您的任务是根据提供的背景信息回答用户的问题。请仔细阅读并记住下面的背景信息，然后根据这些背景信息，\
#     用中文提供准确的结论性回答，但是如果问题的答案不包含在背景信息中，请回答“根据已知信息无法回答该问题，请联系人工客服。”。\
#     在回答之前，请使用“思考链条”来组织您的思考过程，但只需提供思考链条的结论部分。请确保回答的准确性。\n\n\
#         背景信息：问题：ngboh密码忘记，需重置，答案：登录NGBOH系统的网址是[https://starbucks.operations.dynamics.cn/?cmp=SBUX(Report Portal)系统&mi=SBXStoreWorkspace]。在登录页面，使用IAG验证登录方式，即绿围裙登录方式。如果遇到问题，如忘记密码，可以在输入密码下方点击忘记密码来重置。如果仍然无法登录，请联系IT部门电话400-628-7289。\n\n\
#             问题：NGBOH系统登录时显示报错，答案：登录NGBOH系统的网址是[https://starbucks.operations.dynamics.cn/?cmp=SBUX(Report Portal)系统&mi=SBXStoreWorkspace]。在登录页面，使用IAG验证登录方式，即绿围裙登录方式。如果遇到问题，如忘记密码，可以在输入密码下方点击忘记密码来重置。如果仍然无法登录，请联系IT部门电话400-628-7289。\n\n\
#                 回答步骤：\n1. 仔细阅读并保证已经充分理解提供的背景信息。\n2. 使用“思考链条”根据提供的背景信息逐步分析问题，但在回答时仅提供结论。\n3. 如果所需信息在提供的背景信息中，请用中文提供准确的结论性回答。如果所需信息不在提供的背景信息中，请明确指出“根据已知信息无法回答该问题，请联系人工客服。”。\n4. 确保回答的准确性。\n\n\
#                     现在，请根据上述步骤回答用户的问题，要回答的问题是：如何重置NGBOH系统的密码？"
prompt = "您将扮演一个严谨且忠实于所提供的背景信息的客服助手。您的任务是根据提供的背景信息回答用户的问题。请仔细阅读并记住下面的背景信息，然后根据这些背景信息，\
    用中文提供准确的结论性回答，但是如果问题的答案不包含在背景信息中，请回答“根据已知信息无法回答该问题，请联系人工客服。”。\
    在回答之前，请使用“思考链条”来组织您的思考过程，但只需提供思考链条的结论部分。请确保回答的准确性。\n\n\
        背景信息：问题：ngboh无法登录，密码过期，答案：登录NGBOH系统的网址是[https://starbucks.operations.dynamics.cn/?cmp=SBUX(Report Portal)系统&mi=SBXStoreWorkspace]。在登录页面，使用IAG验证登录方式，即绿围裙登录方式。如果遇到问题，如忘记密码，可以在输入密码下方点击忘记密码来重置。如果仍然无法登录，请联系IT部门电话400-628-7289。\n\n\
            问题：问题：ngboh密码忘记，需重置，答案：登录NGBOH系统的网址是[https://starbucks.operations.dynamics.cn/?cmp=SBUX(Report Portal)系统&mi=SBXStoreWorkspace]。在登录页面，使用IAG验证登录方式，即绿围裙登录方式。如果遇到问题，如忘记密码，可以在输入密码下方点击忘记密码来重置。如果仍然无法登录，请联系IT部门电话400-628-7289。n\n\
                回答步骤：\n1. 仔细阅读并保证已经充分理解提供的背景信息。\n2. 使用“思考链条”根据提供的背景信息逐步分析问题，但在回答时仅提供结论。\n3. 如果所需信息在提供的背景信息中，请用中文提供准确的结论性回答。如果所需信息不在提供的背景信息中，请明确指出“根据已知信息无法回答该问题，请联系人工客服。”。\n4. 确保回答的准确性。\n\n\
                    现在，请根据上述步骤回答用户的问题，要回答的问题是：NGBOH系统登录密码过期，如何重置？"


messages = [
    {"role": "system", "content": "你是一个有用的助手。"},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

model_inputs = tokenizer([text], return_tensors="pt").to('cuda')
# 推理前的内存占用
before_inference_gb = convert_bytes_to_gb(torch.cuda.memory_allocated())
generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=512,
    temperature=0.01
)
# 推理后的内存占用
after_inference_gb = convert_bytes_to_gb(torch.cuda.memory_allocated())
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
max_memory_gb = convert_bytes_to_gb(torch.cuda.max_memory_allocated())

print(response)
print(f"Memory usage before inference: {before_inference_gb} GB")
print(f"Memory usage after inference: {after_inference_gb} GB")
print(f"Max memory used during inference: {max_memory_gb} GB")