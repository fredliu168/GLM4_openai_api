# GLM4_openai_api
Glm4 openai api格式接口


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
{"id": "chatcmpl-522bc37b-917d-4f5d-947a-4c113017eb2c", "choices": [{"delta": {"content": "", "response": "", "finished": false}}]}
{"id": "chatcmpl-0238dec6-5c95-485f-88c2-72a8a09169df", "choices": [{"delta": {"content": "你好", "response": "你好", "finished": false}}]}
{"id": "chatcmpl-c06ddfeb-d4f0-4b22-ba12-d98df7a7bf50", "choices": [{"delta": {"content": "，", "response": "你好，", "finished": false}}]}
{"id": "chatcmpl-88ed7d58-ac13-41a7-ac5e-54eb48c28186", "choices": [{"delta": {"content": "我是", "response": "你好，我是", "finished": false}}]}
{"id": "chatcmpl-25d55b09-f583-445b-a2aa-314aa6a7abaa", "choices": [{"delta": {"content": "人工智能", "response": "你好，我是人工智能", "finished": false}}]}
....
["DONE"]
```


使用4bit量化后，占用显存情况：

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


参考以下代码改写：

https://github.com/TylunasLi/ChatGLM-web-stream-demo

https://github.com/NCZkevin/chatglm-web
