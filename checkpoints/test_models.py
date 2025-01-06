# Loading in the needed libraries
import torch
from mini_llama.tokenizer.rust_tokenizer import MiniLlamaTokenizer
from mini_llama.model import MiniLlamaForCausalLM
from datasets import load_dataset

# Loading in the tokenizer configuration
print("Loading in the tokenizer...", end="")

tokenizer_path = "/home/bismarck/mini-llama-3/model/src/tokenizers/tokenizer_configs/pirate_tokenizer_8K.json"
tokenizer = MiniLlamaTokenizer.load(tokenizer_path)

print("Done!")

# Defining and loading in the model
print("Loading in the model...", end="")

model_state_dict = torch.load("/home/bismarck/mini-llama-3/checkpoints/vanilla.pth", weights_only=True)

model = MiniLlamaForCausalLM(
    vocab_size=8192,
    embedding_dim=1024,
    num_decoder_layers=4,
    num_attn_heads=4,
    mlp_layer_intermediate_dim=2048,
    dropout=0.1,
    padding_idx=0,
    tokenizer_path=tokenizer_path
)

# Loading in the weights
model.load_state_dict(model_state_dict)
model = model.to("cuda")
model.eval()

print("Done!")

# Testing out generation
print("Loading in the TinyStories dataset...", end="")
tiny_stories = load_dataset("roneneldan/TinyStories", split="validation")
print("Done!")

# Sample original story
sample = tiny_stories[1]

print("Tokenizing...", end = "")
tokenized_sample = tokenizer.encode("Once upon a time, in a big forest,", 320)
tokenized_sample = torch.tensor(tokenized_sample, dtype=torch.int32)

# Adding the batch dimension
tokenized_sample = tokenized_sample.to("cuda")

print("Done!\n")

# Generating a response
generated_response = tokenizer.decode(model.generate(
    tokenized_sample,
    max_length=320,
    temperature=0.5,
    top_k=1
).cpu().tolist())

# Printing out the generated response
print(generated_response)