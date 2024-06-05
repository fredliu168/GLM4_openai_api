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

参考以下代码改写：

https://github.com/TylunasLi/ChatGLM-web-stream-demo

https://github.com/NCZkevin/chatglm-web
