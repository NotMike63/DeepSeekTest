print("Imports")
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PretrainedConfig
#from auto_gptq import AutoGPTQForCausalLM

# This method checks for the most common torch devices and sets them
def check_device():
    device = None
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device

# defining paths
print("Declaring/Initializing Vars")
DEVICE = check_device()
torch.set_default_device(DEVICE)
project_path = os.getcwd()
target_model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

dataset = ["auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."]

tokenizer = AutoTokenizer.from_pretrained(target_model)
quantized_model = AutoModelForCausalLM.from_pretrained(target_model,
                                                       torch_dtype=torch.bfloat16,
                                                       device_map=DEVICE,
                                                       cache_dir='DeepSeek-R1-Distill-Qwen-1.5B')

print("Model loaded")

# quantized_model.to(DEVICE)
# quantized_model.save_pretrained(model_path)
# quantized_model.generate()

# # Setting up tokenizer
# print("Defining Tokenizer")
# model_inputs = tokenizer([input], return_tensors='pt').to(device)
#
# # loading model
# print("Loading model")
# model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", load_in_4bit=True)
#
# print("Generating model output.")
#  # Generating output tokens
# generated_output = model.generate(**model_inputs)
#
#  # Decoding output tokens
# print("Decoding output tokens")
# tokenizer.batch_decode(generated_output, skip_special_tokens=True)[0]
# tokenizer.pad_token = tokenizer.eos_token
#
# print(tokenizer)
#
# # taking input
# # run 'ollama run deepseek-r1:1.5b'
