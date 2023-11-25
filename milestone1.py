import fire
import json
from llama import Llama

def evaluate_response(model_response: str, correct_answer: str) -> bool:
    final_answer = correct_answer.split('####')[-1].strip()
    print("final_answer: ", final_answer)
    return final_answer.strip() in model_response.strip()

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = None,
    max_gen_len: int = None,
    max_batch_size: int = 1,  # Set batch size to 1
    dataset_file: str = '/home/milesway/research/llama/llama/train.jsonl',
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    # Load math dataset
    with open(dataset_file, 'r') as file:
        problems = [json.loads(line) for line in file][:4]

    total_correct = 0

    # Process and evaluate each problem one by one
    for problem in problems:
        prompt = problem['question']
        correct_answer = problem['answer']

        # Generate model response for the current prompt
        result = generator.text_completion(
            [prompt],  # Pass single prompt
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )[0]
        print(result)

        model_response = result['generation']
        is_correct = evaluate_response(model_response, correct_answer)
        total_correct += int(is_correct)

        # Print problem, model response, and evaluation
        print(f"Problem: {prompt}")
        print(f"Model Response: {model_response}")
        print(f"Correct Answer: {correct_answer}")
        print(f"Correct: {is_correct}")
        print("\n==================================\n")

    # Calculate and print overall accuracy
    accuracy = total_correct / len(problems)
    print(f"Overall Accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    fire.Fire(main)
