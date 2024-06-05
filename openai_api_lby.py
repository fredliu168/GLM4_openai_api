from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel
from typing import Optional
from transformers import AutoTokenizer, AutoModel
import uvicorn
import json
import datetime
import torch
from sse_starlette.sse import ServerSentEvent, EventSourceResponse
import logging
import sys
import uuid
from peft import AutoPeftModelForCausalLM, PeftModelForCausalLM
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
    BitsAndBytesConfig
)

# http://119.91.208.249:10086/project-5/doc-55/
# https://github.com/TylunasLi/ChatGLM-web-stream-demo
# https://github.com/NCZkevin/chatglm-web
DEVICE = "cuda"
DEVICE_ID = "1"
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE
MODEL_DIR = "/home/fred/Documents/llma/model_file/glm-4-9b-chat-1m"
#MODEL_DIR = "/home/fred/Documents/llma/model_file/glm-4-9b-chat"

def getLogger(name, file_name, use_formatter=True):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s    %(message)s')
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

logger = getLogger('ChatGLM', 'chatlog.log')

MAX_HISTORY = 21



class ChatGLM():
    def __init__(self, model_name=MODEL_DIR, quantize_level=4, gpu_id="0,1") -> None:
        logger.info("Start initialize model...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True)
        self.model = self._model(model_name, quantize_level, gpu_id)
        self.model.eval()
        _, _ = self.model.chat(self.tokenizer, "你好", history=[])
        logger.info("Model initialization finished.")
    
    def _model(self, model_name, quantize_level, gpu_id):
        # model_name = "THUDM/chatglm-6b"
        quantize = int(quantize_level)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = None
        if gpu_id == '-1':
            if quantize == 8:
                print('CPU模式下量化等级只能是16或4，使用4')
                #model_name = "THUDM/chatglm-6b-int4"
            elif quantize == 4:
                print('CPU模式下量化等级只能是16或4，使用4')
               #model_name = "THUDM/chatglm-6b-int4"
            #model = AutoModel.from_pretrained(model_name, trust_remote_code=True, device_map='auto').float()
                model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=trust_remote_code, device_map='auto',load_in_4bit=True
        )
        else:
            gpu_ids = gpu_id.split(",")
            self.devices = ["cuda:{}".format(id) for id in gpu_ids]
            if quantize == 16:
                model = AutoModel.from_pretrained(model_name, trust_remote_code=True, device_map='auto').half().cuda()
            else:
                #model = AutoModel.from_pretrained(model_name, trust_remote_code=True, device_map='auto').half().quantize(quantize).cuda()
                #model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map='auto',load_in_4bit=True) #quantization_config=BitsAndBytesConfig(load_in_8bit=True)
                model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map='auto',quantization_config=BitsAndBytesConfig(load_in_4bit=True))
        return model
    
    def clear(self) -> None:
        if torch.cuda.is_available():
            for device in self.devices:
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
    
    def answer(self, query: str, history):
        response, history = self.model.chat(self.tokenizer, query, history=history)
        history = [list(h) for h in history]
        print(response,history)
        return response, history

    def stream(self, query, history,max_length=8192, top_p=0.9,
                                               temperature=0.95):
        if query is None or history is None:
            yield {"query": "", "response": "", "history": [], "finished": True}
        size = 0
        response = ""
        for response, history in self.model.stream_chat(self.tokenizer, query, history, max_length=max_length, top_p=top_p,
                                               temperature=temperature):
            this_response = response[size:]
            history = [list(h) for h in history]
            size = len(response)
            yield {"id":"chatcmpl-"+ str(uuid.uuid4()),"choices":[{"delta":{"content": this_response, "response": response, "finished": False}}]}
        logger.info("Answer - {}".format(response))
        # yield {"query": query, "delta": "[EOS]", "response": response, "history": history, "finished": True}
         
        yield ['DONE']
bot = ChatGLM()

def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()



class ConversationsParams(BaseModel):
    prompt: str
    max_length: Optional[int] = 4096
    top_p: Optional[float] = 0.7
    temperature: Optional[float] = 0.95
    history: Optional[list] = []
    html_entities: Optional[bool] = True
    
origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8080",
    "http://192.168.50.108:10002",
    "http://192.168.50.108:9003",
    "http://qzdm.cn:60011",
    "http://localhost:10002"
]

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/v1/chat/completions-not-stream")
def answer_question(arg_dict: dict):
     
        result = {"query": "", "response": "", "success": False}
        try:
            '''
            {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Hello!"}],
            "stream":true
            }
            '''
            # text = arg_dict["prompt"]
            # ori_history = arg_dict["history"]
            ori_history = []
            print(arg_dict)
            text = ""
            history_qa =()
            length = len(arg_dict["messages"])
            question = ""
            for index, message in enumerate(arg_dict["messages"]):
                
                if index == length -1:
                    text = message["content"]
                    break
                if message["role"] == "user":
                   question = message["content"]
                if message["role"] == "assistant":
                    history_qa = (question,message["content"])
                    
                if index % 2 == 1: 
                    print(history_qa)
                    ori_history.append(history_qa)
                     
            print(ori_history)
            
            
            # text = arg_dict["messages"][0]["content"]
            print(text)
           
            logger.info("Query - {}".format(text))
            # print("Query - {}".format(text))
            if len(ori_history) > 0:
               logger.info("History - {}".format(ori_history))
            #    print("History - {}".format(ori_history))
            history = ori_history[-MAX_HISTORY:]
            history = [tuple(h) for h in history]
            
            text, history = bot.answer(text, history)
            
            ret = {"code":10000,"answer":text, "history": history,"msg":"success"}
            
            return ret
        except Exception as e:
            logger.error(f"error: {e}")
            
            ret={"code":10000,
                   "answer":text, 
                   "history": history,
                   "msg":str(e)
                   }
            return ret

@app.post("/v1/chat/completions")
def answer_question_stream(arg_dict: dict):
        def decorate(generator):
            for item in generator:
                yield ServerSentEvent(json.dumps(item, ensure_ascii=False))
        result = {"query": "", "response": "", "success": False}
        try:
            '''
            {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Hello!"}],
            "stream":true
            }
            '''
            # text = arg_dict["prompt"]
            # ori_history = arg_dict["history"]
            ori_history = []
            print(arg_dict)
            text = ""
            history_qa =()
            length = len(arg_dict["messages"])
            question = ""
            for index, message in enumerate(arg_dict["messages"]):
                
                if index == length -1:
                    text = message["content"]
                    break
                if message["role"] == "user":
                   question = message["content"]
                if message["role"] == "assistant":
                    history_qa = (question,message["content"])
                    
                if index % 2 == 1: 
                    print(history_qa)
                    ori_history.append(history_qa)
                     
            print(ori_history)
            
            
            # text = arg_dict["messages"][0]["content"]
            print(text)
           
            logger.info("Query - {}".format(text))
            # print("Query - {}".format(text))
            if len(ori_history) > 0:
               logger.info("History - {}".format(ori_history))
            #    print("History - {}".format(ori_history))
            if len(ori_history) > MAX_HISTORY:
                history = ori_history[-MAX_HISTORY:]
            else:
                history = ori_history
            # history = ori_history[-MAX_HISTORY:]
            print(len(history),history)   
            history = [tuple(h) for h in history]
            return EventSourceResponse(decorate(bot.stream(text, history)))
        except Exception as e:
            logger.error(f"error: {e}")
            return EventSourceResponse(decorate(bot.stream(None, None)))
        


if __name__ == '__main__':

    uvicorn.run(app, host='0.0.0.0', port=9003, workers=1)
