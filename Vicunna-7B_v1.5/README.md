# LARGE LANGUAGE MODELS FOR SCIENTIFIC PROBLEM SOLVING

This project aims to delve deeper into the concepts of LLM scientific problem solving. By studying representative works, evaluating a known LLM, and enhancing its capabilities by different prompts, students will acquire a comprehensive understanding of current techniques and their limitations.

## Milestone 1

This folder is used for milestone 1, using Vicunna-7B-v1.5 to solve Game 24


## Install

Install all dependency in `requirement.txt`. 

To illustrate, see the command below to run it with the Vicunna-7b model :

```bash
pip install --upgrade openai, torch
pip install "fschat[model_worker,webui]"
```

## Quick Start:

FastChat provides OpenAI-compatible APIs for its supported models, so use FastChat as a local drop-in replacement for OpenAI APIs.

The following OpenAI APIs are supported:
- Chat Completions. (Reference: https://platform.openai.com/docs/api-reference/chat)
- Completions. (Reference: https://platform.openai.com/docs/api-reference/completions)

### RESTful API Server
First, launch the controller

```bash
python3 -m fastchat.serve.controller
```

Then, launch the model worker(s)

```bash
python3 -m fastchat.serve.model_worker --model-path lmsys/vicuna-7b-v1.5
```

Finally, launch the RESTful API server

```bash
python3 -m fastchat.serve.openai_api_server --host localhost --port 8000
```

### OpenAI Official SDK
The goal of `openai_api_server.py` is to implement a fully OpenAI-compatible API server, so the models can be used directly with [openai-python](https://github.com/openai/openai-python) library.

First, install openai-python:
```bash
pip install --upgrade openai
```

Then, interact with model vicuna:
```python
import openai
# to get proper authentication, make sure to use a valid key that's listed in
# the --api-keys flag. if no flag value is provided, the `api_key` will be ignored.
openai.api_key = "EMPTY"
openai.api_base = "http://localhost:8000/v1"

model = "vicuna-7b-v1.5"
prompt = "Once upon a time"

# create a completion
completion = openai.Completion.create(model=model, prompt=prompt, max_tokens=64)
# print the completion
print(prompt + completion.choices[0].text)

# create a chat completion
completion = openai.ChatCompletion.create(
  model=model,
  messages=[{"role": "user", "content": "Hello! What is your name?"}]
)
# print the completion
print(completion.choices[0].message.content)
```

Streaming is also supported. See [test_openai_api.py](../tests/test_openai_api.py).  If your api server is behind a proxy you'll need to turn off buffering, you can do so in Nginx by setting `proxy_buffering off;` in the location block for the proxy.

### Test for Milestone 1
 -  [`Game 24`](https://www.4nums.com/game/difficulties/)

    - Game of 24 is a mathematical reasoning challenge, where the goal is to use 4 numbers and basic arithmetic operations (+-*/) to obtain 24. For example, given input “4 9 10 13”, a solution output
􏰱􏰮􏰧􏰲 􏰀􏰡􏰳􏰳 􏱅􏰧􏰥􏱇􏱅􏱈 􏱜􏰡􏱈􏱏 􏰤􏰥􏰦􏰥􏰧
could be “(10 - 4) * (13 - 9) = 24”.

    - Expected value 0%.

### Fine-tuned Chat Models

The fine-tuned models were trained for dialogue applications. To get the expected features and performance for them, a specific formatting defined in [`chat_completion`](https://github.com/facebookresearch/llama/blob/main/llama/generation.py#L212)
needs to be followed, including the `INST` and `<<SYS>>` tags, `BOS` and `EOS` tokens, and the whitespaces and breaklines in between (we recommend calling `strip()` on inputs to avoid double-spaces).

You can also deploy additional classifiers for filtering out inputs and outputs that are deemed unsafe.

Examples using llama-2-7b-chat:

```
torchrun --nproc_per_node 1 example_chat_completion.py \
    --ckpt_dir llama-2-7b-chat/ \
    --tokenizer_path tokenizer.model \
    --max_seq_len 512 --max_batch_size 6
```

### Inference

Different models require different model-parallel (MP) values:

|  Model | MP |
|--------|----|
| 7B     | 1  |
| 13B    | 2  |
| 70B    | 8  |



## References

1. [Research Paper](https://ai.meta.com/research/publications/llama-2-open-foundation-and-fine-tuned-chat-models/)
2. [Llama 2 technical overview](https://ai.meta.com/resources/models-and-libraries/llama)
3. [Open Innovation AI Research Community](https://ai.meta.com/llama/open-innovation-ai-research-community/)

