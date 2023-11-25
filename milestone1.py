# Import necessary libraries
import fire
import json
from llama import Llama

def evaluate_response(model_response: str, correct_answer: str) -> bool:
    """
    Evaluate the model response against the correct answer.

    Args:
        model_response (str): The response generated by the model.
        correct_answer (str): The correct answer for comparison.

    Returns:
        bool: True if the model response contains the correct answer, False otherwise.
    """
    # Extract the final part of the correct answer after '####' and strip whitespace
    final_answer = correct_answer.split('####')[-1].strip()
    print("final_answer: ", final_answer)
    # Check if the final answer is in the model response
    return final_answer.strip() in model_response.strip()

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = None,
    max_gen_len: int = None,
    max_batch_size: int = 1,  # Default batch size set to 1
    dataset_file: str = '/home/milesway/research/llama/llama/train.jsonl',
):
    """
    Main function to initialize the model, load dataset, generate responses, and evaluate.

    Args:
        ckpt_dir (str): Directory of the checkpoint.
        tokenizer_path (str): Path of the tokenizer.
        temperature (float, optional): Temperature parameter for text generation. Defaults to 0.6.
        top_p (float, optional): Top-p parameter for text generation. Defaults to 0.9.
        max_seq_len (int, optional): Maximum sequence length for the model. Defaults to None.
        max_gen_len (int, optional): Maximum generation length. Defaults to None.
        max_batch_size (int, optional): Maximum batch size for processing. Defaults to 1.
        dataset_file (str, optional): Path to the dataset file. Defaults to '/home/milesway/research/llama/llama/train.jsonl'.

    """
    # Initialize the Llama model with provided arguments
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    # Load the dataset from the specified file and read the first 4 problems
    with open(dataset_file, 'r') as file:
        problems = [json.loads(line) for line in file][:4]

    total_correct = 0

    # Process and evaluate each problem
    for problem in problems:
        prompt = problem['question']
        correct_answer = problem['answer']

        # Generate response from the model for the given prompt
        result = generator.text_completion(
            [prompt],  # Input single prompt
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )[0]
        print(result)

        model_response = result['generation']
        is_correct = evaluate_response(model_response, correct_answer)
        total_correct += int(is_correct)

        # Print problem, response, and evaluation result
        print(f"Problem: {prompt}")
        print(f"Model Response: {model_response}")
        print(f"Correct Answer: {correct_answer}")
        print(f"Correct: {is_correct}")
        print("\n==================================\n")

    # Calculate and display the overall accuracy
    accuracy = total_correct / len(problems)
    print(f"Overall Accuracy: {accuracy:.2f}")

# Execute the main function using fire for command-line interface interaction
if __name__ == "__main__":
    fire.Fire(main)
