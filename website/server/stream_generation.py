# Importing the needed libraries
import torch
from mini_llama.model import MiniLlamaForCausalLM
from mini_llama.tokenizer.rust_tokenizer import MiniLlamaTokenizer

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import asyncio

from typing import List

# Defining the FastAPI app
app = FastAPI()

origins = [
    "http://localhost",
    "http://20.94.238.87:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_config = {
    "vocab_size": 8192,
    "embedding_dim": 1024,
    "num_decoder_layers": 4,
    "num_attn_heads": 4,
    "mlp_layer_intermediate_dim": 2048,
    "dropout": 0.1,
    "padding_idx": 0,
    "tokenizer_path": "/home/bismarck/mini-llama-3/model/src/tokenizers/tokenizer_configs/pirate_tokenizer_8K.json"
}

# Fetching in the Rust tokenizer!
tokenizer = MiniLlamaTokenizer.load(model_config["tokenizer_path"])

# Loading in the vanilla TinyStories LLM :)
vanilla_model = MiniLlamaForCausalLM(**model_config)

vanilla_state_dict = torch.load("/home/bismarck/mini-llama-3/checkpoints/vanilla.pth", weights_only=True)
vanilla_model.load_state_dict(vanilla_state_dict)
vanilla_model = vanilla_model.to("cuda")
vanilla_model.eval()

# Loading in the finetuned blackbeard LLM!
blackbeard_model = MiniLlamaForCausalLM(**model_config)

blackbeard_state_dict = torch.load("/home/bismarck/mini-llama-3/checkpoints/blackbeard.pth", weights_only=True)
blackbeard_model.load_state_dict(blackbeard_state_dict)
blackbeard_model = blackbeard_model.to("cuda")
blackbeard_model.eval()


# Defining an asynchronous generation function to which we can specify a model type!
async def generate_stream(model: MiniLlamaForCausalLM, temperature: float, top_k: int, input_ids: List[int]):
    """Generates a stream of the model outputs

    Args:
        model: MiniLlamaForCausalLM: The model with which to generate inference from
        temperature (float): Controls the variability of the model -- The logits are divided with this value :)
        top_k (int): The number of top K most likely tokens to sample a token from with a multinomial setup
        input_ids (List[int]): A list of tokenized inputs
    
    Yields:
        Returns each token as it is generated!
    """
    
    # Being extra safe here :)
    try:
        
        for token_id in model.streaming_generate(input_ids=input_ids, temperature=temperature, top_k=top_k):
            
            # Here we convert the token back into the string representation before passing
            yield f"{tokenizer.decode([token_id])}"
            await asyncio.sleep(0)
            
    except Exception as e:
        print(f"An error occured! \n\n{e}")
        
@app.post("/generation")
async def vanilla_stream_api(request: Request):
    
    try:
        
        # Fetching the prompt from the request
        data = await request.json()

        # Fetching the model configurations and the prompt
        model_str = data.get("model", "vanilla")
        temperature = data.get("temperature", 0.75)
        top_K = data.get("top_k", 8)
        prompt = data.get("prompt", "")
        
        # Converting the prompt into tokens
        input_ids = tokenizer.encode(prompt, 320)
        input_ids = torch.tensor(input_ids, dtype=torch.int32).cuda()
        
        # Choosing the LLM
        if model_str == "vanilla":
            model = vanilla_model
            
        else:
            model = blackbeard_model
            
        # Returning a StreamingResponse object for a smoother experience! :)
        return StreamingResponse(generate_stream(model, temperature, top_K, input_ids), media_type="text/plain")
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    