import torch
import os
import sys
import time
import random
import argparse
root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)

from transformers import AutoTokenizer
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser(description="Generation benchmarking")
parser.add_argument("--prompt", type=str, default="The future of AI is")
parser.add_argument("--promptlen", type=int, default=100)
parser.add_argument("--genlen", type=int, default=200)
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--topk", type=int, default=1)
parser.add_argument("--topp", type=float, default=1.0)
parser.add_argument("--minp", type=float, default=0.0)
parser.add_argument("--repetition-penalty", type=float, default=1.0)
parser.add_argument("--batch", type=int, default=1)
args = parser.parse_args()

# Load the trained model
repeats = 100
model_path = "mamba_model"  # Change this if you saved the model in a different directory
model = MambaLMHeadModel.from_pretrained(model_path, device = DEVICE)
model.eval()  # Set model to evaluation mode
print(model)
print(f"Number of parameters: {(sum(p.numel() for p in model.parameters() if p.requires_grad))/ 1e6:.2f}M")


# Load the tokenizer (assuming you used a Hugging Face tokenizer during training)
tokenizer = AutoTokenizer.from_pretrained("gpt2")  # Replace "gpt2" if you used a different tokenizer
###########
tokens = tokenizer(args.prompt, return_tensors="pt")
input_ids = tokens.input_ids.to(device=DEVICE)
attn_mask = tokens.attention_mask.to(device=DEVICE)

# input_text = "the future of AI is"
# input_ids = tokenizer.encode(input_text, return_tensors="pt").to(DEVICE)

max_length = input_ids.shape[1] + args.genlen
with torch.no_grad():
       
        generated_ids = model.generate(
            input_ids=input_ids,
            max_length=max_length,
            cg=True,
            return_dict_in_generate=True,
            output_scores=True,
            enable_timing=False,
            temperature=args.temperature,
            top_k=args.topk,
            top_p=args.topp,
            min_p=args.minp,
            repetition_penalty=args.repetition_penalty,
        )

start = time.time()
for _ in range(repeats):
    with torch.no_grad():
        # with torch.autograd.profiler.profile(use_cuda=True) as prof:
        generated_ids = model.generate(
            input_ids=input_ids,
            max_length=max_length,
            cg=True,
            return_dict_in_generate=True,
            output_scores=True,
            enable_timing=False,
            temperature=args.temperature,
            top_k=args.topk,
            top_p=args.topp,
            min_p=args.minp,
            repetition_penalty=args.repetition_penalty,
        )

        # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

# Calculate the generation time
end = time.time()
generation_time = (end - start)/repeats * 1000
generated_text = tokenizer.decode(generated_ids.sequences[0].tolist(), skip_special_tokens=True)

print(f"Generation Time: {generation_time:.4f} milli seconds")
print(f"Prompt Length: {len(input_ids[0])}")
print(f"Output Sequence Length: {len(generated_ids.sequences[0])}")
print(generated_text)
