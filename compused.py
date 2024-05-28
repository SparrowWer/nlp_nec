from vllm import LLM, SamplingParams
import time

# Sample prompts.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Create an LLM.
llm = LLM(model="/data0/zhangchongrui/model/compused",trust_remote_code=True, tensor_parallel_size=2,gpu_memory_utilization=0.98,max_model_len=2048)
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
#起始时间
start = time.perf_counter()
total_tokens = 0
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    total_tokens += len(generated_text.split())  # count the number of tokens in the generated text
end = time.perf_counter()
print("运行时间：", end-start, "秒")
print("Average generate speed (tokens/s): {}".format(total_tokens / (end-start)))