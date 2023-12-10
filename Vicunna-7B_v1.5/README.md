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

### Run our code

For simple prompting, run the following command:

```bash
python scientific_llm/Vicunna-7B_v1.5/milestone_1_Vicunna_simple.py
```

For strightfoward prompting, run the following command:

```bash
python scientific_llm/Vicunna-7B_v1.5/milestone_1_Vicunna_strightforward.py
```

## Test for Milestone 1
 -  [`Game 24`](https://www.4nums.com/game/difficulties/)

    - Game of 24 is a mathematical reasoning challenge, where the goal is to use 4 numbers and basic arithmetic operations (+-*/) to obtain 24. For example, given input “4 9 10 13”, a solution output could be “(10 - 4) * (13 - 9) = 24”

    - Expected value 0%.

