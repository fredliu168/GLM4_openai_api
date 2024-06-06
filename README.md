# GLM4_openai_api
Glm4 openai api格式接口,目前还不支持函数调用


/POST http://127.0.0.1:8003/v1/chat/completions
```
{
"model":"glm4",
"stream":true,
"messages":[
{"role":"system", "content": "you are a helpful assistant"},
{"role":"user","content":"你好，介绍一下你自己"}]
}
```

返回：
```
["DONE"]
12:25:36
{"model": "glm4", "id": "chatcmpl-b86cd623-e83e-474e-94b1-5ae9734f85e0", "object": "chat.completion.chunk", "choices": [{"delta": {"role": "assistant", "content": "", "function_call": null}, "finish_reason": "stop", "index": 0}]}
12:25:36
{"model": "glm4", "id": "chatcmpl-9078816b-40f1-4937-839d-508a7fb17f4f", "object": "chat.completion.chunk", "choices": [{"delta": {"role": "assistant", "content": "。", "function_call": null}, "finish_reason": "", "index": 0}]}
12:25:36
{"model": "glm4", "id": "chatcmpl-7a9da6ae-85bf-455f-b31c-700a02f5b02c", "object": "chat.completion.chunk", "choices": [{"delta": {"role": "assistant", "content": "和支持", "function_call": null}, "finish_reason": "", "index": 0}]}
12:25:36

```
```
{
"model":"glm4",
"stream":false,
"messages":[
{"role":"system", "content": "you are a helpful assistant"},
{"role":"user","content":"你好，介绍一下你自己"}]
}
```
```
{
    "model": "glm4",
    "id": "6860e447-a046-4fcd-b73d-d5cc96f78728",
    "object": "chat.completion.chunk",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "你好，我是人工智能助手智谱清言（ChatGLM），是基于智谱 AI 公司训练的语言模型 GLM-4 模型开发的，我的目标是针对用户的问题和要求提供适当的答复和支持。",
                "function_call": null
            },
            "finish_reason": "stop"
        }
    ]
}
```

使用4bit量化后，占用显存情况：
```
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.171.04             Driver Version: 535.171.04   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA GeForce RTX 2080 Ti     Off | 00000000:01:00.0 Off |                  N/A |
| 22%   48C    P2              62W / 250W |   3926MiB / 11264MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
|   1  NVIDIA GeForce GTX 1070        Off | 00000000:02:00.0 Off |                  N/A |
|  0%   49C    P2              52W / 230W |   5460MiB /  8192MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
```

参考以下代码改写：

https://github.com/TylunasLi/ChatGLM-web-stream-demo

https://github.com/NCZkevin/chatglm-web
