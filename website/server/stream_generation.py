# Importing the needed libraries
import torch
from mini_llama.model import MiniLlamaForCausalLM
from mini_llama.tokenizer.rust_tokenizer import MiniLlamaTokenizer

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
import asyncio

from typing import List

# Defining the FastAPI app
app = FastAPI()

# Defining the needed objects
tokenizer_path = "/home/bismarck/mini-llama-3/model/src/tokenizers/tokenizer_configs/pirate_tokenizer_8K.json"
tokenizer = MiniLlamaTokenizer.load(tokenizer_path)

# Defining the model architecture and loading in the weights
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

# Defining an asynchronous generation function
async def generate_stream(input_ids: List[int]):
    """Generates a stream of the model outputs

    Args:
        input_ids (List[int]): A list of tokenized inputs
    """
    
    # Being extra safe here :)
    try:
        for token_id in model.streaming_generate(input_ids):
            
            # Here we convert the token back into the string representation before passing
            yield f"{tokenizer.decode([token_id])}"
            await asyncio.sleep(0)
            
    except Exception as e:
        print(f"An error occured! \n\n{e}")
        
@app.post("/generate")
async def generate_stream_api(request: Request):
    
    try:
        
        # Fetching the prompt from the request
        data = await request.json()
        prompt = data.get("prompt", "")

        # Converting the prompt into tokens
        input_ids = tokenizer.encode(prompt, 320)
        input_ids = torch.tensor(input_ids, dtype=torch.int32).cuda()
        
        # Returning a StreamingResponse object for a smoother experience! :)
        return StreamingResponse(generate_stream(input_ids), media_type="text/plain")
    
    except Exception as e:
        
        return JSONResponse(content={"error": str(e)}, status_code=500)
    
    