import torch
import os
import argparse
import uvicorn
import time
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Optional
from contextlib import asynccontextmanager
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model = None
tokenizer = None
CHECKPOINT_PATH = "" 

def get_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Run an OpenAI-compatible FastAPI server for a local LLM.")
    parser.add_argument(
        "-c", "--checkpoint-path", 
        type=str, 
        default="", 
        help="Hugging Face model checkpoint path (local or on Hub)"
    )
    # Standard local defaults
    parser.add_argument("--server-port", type=int, default=8000, help="Server port")
    parser.add_argument("--server-name", type=str, default="127.0.0.1", help="Server name or IP address")
    return parser.parse_args()

def load_model_and_tokenizer(checkpoint_path: str):
    """Loads the pre-trained model and tokenizer from a path."""
    if not os.path.isdir(checkpoint_path):
        print(f"📥 Model not found locally at '{checkpoint_path}', attempting to download from Hub...")

    print(f"🚀 Loading model and tokenizer from {checkpoint_path}...")
    
    device_map = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"✅ Using device: {device_map}")

    local_tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
    
    local_model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype="auto",
        device_map=device_map,
        trust_remote_code=True
    ).eval()
    
    print("✅ Model and tokenizer loaded successfully.")
    return local_model, local_tokenizer

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer
    model, tokenizer = load_model_and_tokenizer(CHECKPOINT_PATH)
    yield
    model = None
    tokenizer = None

app = FastAPI(lifespan=lifespan)
os.environ["TRUST_REMOTE_CODE"] = "true" 

# --- OpenAI-compatible Pydantic Models ---
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str 
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 2048
    temperature: Optional[float] = 0.7 

class Choice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str = "stop"

class ChatCompletionResponse(BaseModel):
    id: str = "chatcmpl-local"
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[Choice]

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    """Handles chat completion requests compatible with the OpenAI API."""
    prompt = tokenizer.apply_chat_template(
        [{"role": m.role, "content": m.content} for m in request.messages],
        tokenize=False,
        add_generation_prompt=True
    )
    
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    
    outputs = pipe(
        prompt,
        max_new_tokens=request.max_tokens,
        do_sample=True,
        temperature=request.temperature if request.temperature > 0 else 0.01,
        top_p=0.95,
        pad_token_id=pipe.tokenizer.eos_token_id
    )
    
    generated_text = outputs[0]['generated_text']
    response_text = generated_text.split(prompt)[-1].strip()

    response_message = ChatMessage(role="assistant", content=response_text)
    choice = Choice(index=0, message=response_message)
    
    return ChatCompletionResponse(model=CHECKPOINT_PATH, choices=[choice])

if __name__ == "__main__":
    args = get_args()
    
    CHECKPOINT_PATH = args.checkpoint_path
    
    print(f"Starting server on {args.server_name}:{args.server_port} to mimic OpenAI API")
    
    uvicorn.run(app, host=args.server_name, port=args.server_port)