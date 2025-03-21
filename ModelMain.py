print("Imports")
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
#from auto_gptq import AutoGPTQForCausalLM


# defining paths
print("Declaring/Initializing Vars")
device = "mps"
project_path = os.getcwd()
target_model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer_path = os.path.join(project_path, target_model, "tokenizer.json")
quantized_model_path = os.path.join(project_path, target_model, "GPTQ")
model_path = os.path.join(project_path, target_model, "model.safetensors")
#device, __, __ = get_backend()
dataset = ["auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."]

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=target_model)
gptq_config = GPTQConfig(bits=4, dataset=dataset, tokenizer=tokenizer)
quantized_model = AutoModelForCausalLM.from_pretrained(model_path, device_map="mps", quantization_config=gptq_config)

quantized_model.to(device)
quantized_model.save_pretrained(model_path)
quantized_model.generate()

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