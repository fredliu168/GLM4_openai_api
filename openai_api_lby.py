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
#https://github.com/fredliu168/GLM4_openai_api
# Constants
MODEL_DIR = "/home/glm-4-9b-chat-1m"
MAX_HISTORY = 21
OUT_MAX_LEN = 8192

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
    
    def answer(self, query: str, history: List[tuple]) -> (str, List[tuple]):
        response, history = self.model.chat(self.tokenizer, query, history=history)
        return response, [list(h) for h in history]

    def stream(self, query: str, history: List[tuple], max_length: int = OUT_MAX_LEN, top_p: float = 0.9, temperature: float = 0.95):
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
                        "finish_reason": "",
                        "index": 0
                    }
                ]
            }
            
        yield {
                "model": "glm4",
                "id": "chatcmpl-" + str(uuid.uuid4()),
                "object": "chat.completion.chunk",
                "choices": [
                    {
                        "delta": {
                            "role": "assistant",
                            "content": "",
                            "function_call": None
                        },
                        "finish_reason": "stop",
                        "index": 0
                    }
                ]
            }
        yield ["DONE"]

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
    
    try:
        ori_history = []
        messages = arg_dict.get("messages", [])
        text = messages[-1]["content"] if messages else ""
        history_qa = ()
        
        for index, message in enumerate(messages[:-1]):
            if message["role"] == "user":
                question = message["content"]
            elif message["role"] == "assistant" and message["role"] == "system":
                history_qa = (question, message["content"])
            if index % 2 == 1:
                ori_history.append(history_qa)
        
        #logger.info(f"Query - {text}")
        #if ori_history:
        #   logger.info(f"History - {ori_history}")
        
        history = ori_history[-MAX_HISTORY:]
        history = [tuple(h) for h in history]
        
        if arg_dict.get("stream", False):
            return EventSourceResponse(decorate(bot.stream(text, history)))
        else:
            response, history = bot.answer(text, history)
            return {
                "model": "glm4",
                "id": str(uuid.uuid4()),
                "object": "chat.completion.chunk",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": response,
                            "function_call": None
                        },
                        "finish_reason": "stop"
                    }
                ]
            }
    except Exception as e:
        logger.error(f"error: {e}")
        return {
            "model": "glm4",
            "id": str(uuid.uuid4()),
            "object": "chat.completion.chunk",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "function_call": None
                    },
                    "finish_reason": "stop"
                }
            ],
            "msg": str(e)
        }

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=9003, workers=1)
