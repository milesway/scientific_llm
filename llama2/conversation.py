# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import List, Optional

import fire

from llama import Llama, Dialog

def converse_with_llama(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 8,
    max_gen_len: Optional[int] = None,
):
    """
    Interactive conversation with the Llama model.
    """
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    dialog: List[Dialog] = []

    print("You can start the conversation. Type 'exit' to end.")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("Goodbye!")
            break

        dialog.append({"role": "user", "content": user_input})
        result = generator.chat_completion(
            [dialog], 
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )[0]
        
        # Add the generated content to the dialog history
        dialog.append(result['generation'])
        
        print(f"Assistant: {result['generation']['content']}")

if __name__ == "__main__":
    # You can call the function directly or use fire for command line interfaces
    # fire.Fire(converse_with_llama)


    from transformers import AutoTokenizer, AutoModelForCausalLM
    import transformers
    import torch

    checkpoint = "PygmalionAI/pygmalion-2-7b"


    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, device_map="auto", offload_folder="offload", torch_dtype=torch.float16
)


    converse_with_llama(
        ckpt_dir="path_to_checkpoint",
        tokenizer_path="path_to_tokenizer"
    )
