# LARGE LANGUAGE MODELS FOR SCIENTIFIC PROBLEM SOLVING

This project aims to delve deeper into the concepts of LLM scientific problem solving. By studying representative works, evaluating a known LLM, and enhancing its capabilities by different prompts, students will acquire a comprehensive understanding of current techniques and their limitations.

## Setting up the environment
In order to setup the development environment, we have provided an environment.yml file to quickly install all the required packages. Run the following command to create it.
```
conda env create -f environment.yml
```
After that run the following command to activate the environment
```
conda activate llm
```


## Milestone

Each folder has detailed comments for running the code. Please refer to the README in folder.

- Milestone1:
    
    * [`Grad School Math`](https://github.com/openai/grade-school-math) with Llama2 (https://huggingface.co/meta-llama/Llama-2-7b)

        * GSM8K consists of 8.5K high quality grade school math problems created by human problem writers. These problems take between 2 and 8 steps to solve, and solutions primarily involve performing a sequence of elementary calculations using basic arithmetic operations (+ - / *) to reach the final answer. A bright middle school student should be able to solve every problem.

        * Expected value 14.6% accuary.
    
    * [`Game 24`](https://www.4nums.com/game/difficulties/) with Vicuna-7B

        * Game of 24 is a mathematical reasoning challenge, where the goal is to use 4 numbers and basic arithmetic operations (+-*/) to obtain 24. For example, given input “4 9 10 13”, a solution output could be “(10 - 4) * (13 - 9) = 24”.

        * Expected value 0%. 

- Milestone2:
   * In order to reproduce our results for milestone 2, you can simply use the prompts shown in our report and use that as input to the Vicuna-7B model. One example is: *Use basic arithmetic operations (+ - * /) to obtain 24 from the following 4 numbers step by step. For each step, you are only allowed to choose two of the remaining numbers to obtain a new number. Input: 4, 5, 6, 10*
   * Expected value close to 0%.

- Milestone3:
  * Our results for milestone 3 can be simply reproduced by running the following command
    ```
    python 24.py --n_generate_sample=5 --method_select=greedy
    ```
