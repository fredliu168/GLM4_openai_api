from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import uvicorn
import json
import torch
from sse_starlette.sse import ServerSentEvent, EventSourceResponse
import logging
import sys
import uuid

# Constants
MODEL_DIR = "/home/lby/llm/model/glm-4-9b-chat-1m"
MAX_HISTORY = 21
MAX_LENGTH = 8192
TOP_P = 0.8
TEMPERATURE = 0.8

# Logger setup
def get_logger(name: str, file_name: str, use_formatter: bool = True) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    if file_name:
        handler = logging.FileHandler(file_name, encoding='utf8')
        handler.setLevel(logging.INFO)
        if use_formatter:
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
            handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

logger = get_logger('ChatGLM', 'chatlog.log')

class ChatGLM:
    def __init__(self, model_name: str = MODEL_DIR) -> None:
        logger.info("Start initialize model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = self._load_model(model_name)
        self.model.eval()      
        logger.info("Model initialization finished.")
    
    def _load_model(self, model_name: str):
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map='auto', quantization_config=BitsAndBytesConfig(load_in_4bit=True))
        return model

    def clear(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    
    def answer(self, query: str, history: List[tuple],max_length: int = 81920, top_p: float = 0.9, temperature: float = 0.95) -> (str, List[tuple]):
        response, history = self.model.chat(self.tokenizer, query, history=history, max_length=max_length, top_p=top_p, temperature=temperature)
        return response, [list(h) for h in history]

    def stream(self, query: str, history: List[tuple], max_length: int = 81920, top_p: float = 0.9, temperature: float = 0.95):
        size = 0
        response = ""
        for response, history in self.model.stream_chat(self.tokenizer, query, history, max_length=max_length, top_p=top_p, temperature=temperature):
            this_response = response[size:]
            size = len(response)
            yield {
                "model": "glm4",
                "id": "chatcmpl-" + str(uuid.uuid4()),
                "object": "chat.completion.chunk",
                "choices": [
                    {
                        "delta": {
                            "role": "assistant",
                            "content": this_response,
                            "function_call": None
                        },
                        "finish_reason": "length",
                        "index": 0
                    }
                ]
            }       
        

bot = ChatGLM()


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/v1/chat/completions")
async def completions(arg_dict: Dict[str, Any]):
    def decorate(generator):
        for item in generator:
            yield ServerSentEvent(json.dumps(item, ensure_ascii=False))
        yield ServerSentEvent(data="[DONE]")
    try:       
        print(arg_dict)
        messages = arg_dict.get("messages", [])
        text = messages[-1]["content"] if messages else ""
        history = []
        if len(messages)>1:
            history = messages[:-1]
            if len(history)>MAX_HISTORY:
                history = messages[0]+history[-MAX_HISTORY:]      
         
        top_p = arg_dict.get("top_p",TOP_P)
        temperature = arg_dict.get("temperature",TEMPERATURE)
        max_length = arg_dict.get("max_tokens",MAX_LENGTH)
        
        if max_length < 1024:
            max_length = TEMPERATURE
        
        if temperature == 0:
           temperature = 0.1 
        
        if arg_dict.get("stream", False):
            return EventSourceResponse(decorate(bot.stream(text, history,top_p = top_p,temperature = temperature,max_length = max_length)))
        else:
            response, history = bot.answer(text, history)
            return {
                "model": "glm4",
                "id": str(uuid.uuid4()),
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": response,
                            "function_call": None
                        },
                        "finish_reason": "length"
                    }
                ]
            }
    except Exception as e:
        logger.error(f"error: {e}")
        return {
            "model": "glm4",
            "id": str(uuid.uuid4()),
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "function_call": None
                    },
                    "finish_reason": "length"
                }
            ],
            "msg": str(e)
        }

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8003, workers=1)
