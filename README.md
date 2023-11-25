# LARGE LANGUAGE MODELS FOR SCIENTIFIC PROBLEM SOLVING

This project aims to delve deeper into the concepts of LLM scientific problem solving. By studying representative works, evaluating a known LLM, and enhancing its capabilities by different prompts, students will acquire a comprehensive understanding of current techniques and their limitations.

## Download

In order to download the model weights and tokenizer, please visit the [Meta website](https://ai.meta.com/resources/models-and-libraries/llama-downloads/) and accept License.

Once request is approved, run the download.sh script, passing the URL provided when prompted to start the download.

Pre-requisites: Make sure you have `wget` and `md5sum` installed. Then to run the script: `./download.sh`.


## Quick Start

See `example_text_completion.py` for some examples. 

To illustrate, see the command below to run it with the llama-2-7b model (`nproc_per_node` needs to be set to the `MP` value):

```
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 milestone1.py     --ckpt_dir llama-2-7b/     --tokenizer_path tokenizer.model     --max_seq_len 128 --max_batch_size 1
```

## Inference

Different models require different model-parallel (MP) values:

|  Model | MP |
|--------|----|
| 7B     | 1  |
| 13B    | 2  |
| 70B    | 8  |

### Fine-tuned Chat Models

The fine-tuned models were trained for dialogue applications. To get the expected features and performance for them, a specific formatting defined in [`chat_completion`](https://github.com/facebookresearch/llama/blob/main/llama/generation.py#L212)
needs to be followed, including the `INST` and `<<SYS>>` tags, `BOS` and `EOS` tokens, and the whitespaces and breaklines in between (we recommend calling `strip()` on inputs to avoid double-spaces).

You can also deploy additional classifiers for filtering out inputs and outputs that are deemed unsafe. See the llama-recipes repo for [an example](https://github.com/facebookresearch/llama-recipes/blob/main/inference/inference.py) of how to add a safety checker to the inputs and outputs of your inference code.

Examples using llama-2-7b-chat:

```
torchrun --nproc_per_node 1 example_chat_completion.py \
    --ckpt_dir llama-2-7b-chat/ \
    --tokenizer_path tokenizer.model \
    --max_seq_len 512 --max_batch_size 6


## References

1. [Research Paper](https://ai.meta.com/research/publications/llama-2-open-foundation-and-fine-tuned-chat-models/)
2. [Llama 2 technical overview](https://ai.meta.com/resources/models-and-libraries/llama)
3. [Open Innovation AI Research Community](https://ai.meta.com/llama/open-innovation-ai-research-community/)

