import itertools
import numpy as np
from functools import partial
from tot.models import gpt
import copy

def get_current_numbers(y: str) -> str:
    last_line = y.strip().split('\n')[-1]
    return last_line.split('left: ')[-1].split(')')[0]

def get_value(task, x, y, n_evaluate_sample, cache_value=True):
    value_prompt = task.value_prompt_wrap(x, y)
    if cache_value and value_prompt in task.value_cache:
        return task.value_cache[value_prompt]
    value_outputs = gpt(value_prompt, n=n_evaluate_sample, stop=None)
    value = task.value_outputs_unwrap(x, y, value_outputs)
    if cache_value:
        task.value_cache[value_prompt] = value
    return value

def get_values(task, x, ys, n_evaluate_sample, cache_value=True):
    values = []
    local_value_cache = {}
    for y in ys:  # each partial output
        if y in local_value_cache:  # avoid duplicate candidates
            value = 0
        elif y == '24':
            value = 1.0
        else:    
            value = get_value(task, x, y, n_evaluate_sample, cache_value=cache_value)
            local_value_cache[y] = value
        values.append(value)
    return values

def get_votes(task, x, ys, n_evaluate_sample):
    vote_prompt = task.vote_prompt_wrap(x, ys)
    vote_outputs = gpt(vote_prompt, n=n_evaluate_sample, stop=None)
    values = task.vote_outputs_unwrap(vote_outputs, len(ys))
    return values

def get_proposals(task, x, y):
    if y == '24':
        return [y]
    if y:
        x = get_current_numbers(y)

    print("current x:", x)
    propose_prompt = task.propose_prompt_wrap(x, y)
    # change 1 to 5 to get more initial steps to start with
    proposals = []
    # for i in range(5):
    proposals += gpt(propose_prompt, n=1, stop=None)[0].split('\n')
    
    # dedup proposals
    proposals = list(set(proposals))
    # trim proposals.
    trimed_proposals = trim(x,proposals)
    # fix the set.
    fixed_proposals = fix_left(x, trimed_proposals)
    return [y + _ + '\n' for _ in fixed_proposals]

def fix_left(x, proposed_y):
    """
    The model often is not able to perform set union or intersection, again we utilize python to do it.
    This function is to fix the elements that's left after one step.
    """
    split_x = x.split(' ')
    y_ok = []
    for proposal in proposed_y:
        try:
            equation, left = proposal.split('(')
            # equation is like 4 + 5 = 9. 4 and 5 should be in x
            equation_list = equation.split(' ')
            operator1 = equation_list[0]
            operator2 = equation_list[2]
            result = equation_list[4]
            prev_x = [] + split_x
            prev_x.remove(operator1)
            prev_x.remove(operator2)

            prev_x.append(result)
            sorted(prev_x)
            y_ok.append(equation + "(left: " + (" ".join(prev_x)) + ")")
        except:
            # in some cases, the model response is incorrect, we justs skip them.
            pass
    return y_ok



def trim(x, proposed_y):
    """
    trim bad responses, for example the starting input is 4,5,6,10, the model outputs 6 + 1 = 7 (left: 7 6 10) 
    among which 1 is not part of the input.
    """
    split_x = x.split(' ')
    y_ok = []
    for proposal in proposed_y:
        try:
            equation, left = proposal.split('(')
            # equation is like 4 + 5 = 9. 4 and 5 should be in x
            equation_list = equation.split(' ')
            operator1 = equation_list[0]
            operator2 = equation_list[2]
            result = equation_list[4]
            # if the operator is not from the input list, we should discard it.
            if operator1 not in split_x or operator2 not in split_x:
                continue
            left_split = left.split(" ")
            # if the added result is not in the left (left: 6 9 10)
            if result not in left_split:
                continue
            y_ok.append(proposal)
        except:
            # in some cases, the model response is incorrect, we justs skip them.
            pass
    return y_ok







def get_samples(task, x, y, n_generate_sample, prompt_sample, stop):
    if prompt_sample == 'standard':
        prompt = task.standard_prompt_wrap(x, y)
    elif prompt_sample == 'cot':
        prompt = task.cot_prompt_wrap(x, y)
    else:
        raise ValueError(f'prompt_sample {prompt_sample} not recognized')
    samples = gpt(prompt, n=n_generate_sample, stop=stop)
    return [y + _ for _ in samples]

def solve(args, task, idx, to_print=True):
    global gpt
    gpt = partial(gpt, model=args.backend, temperature=args.temperature)
    print(gpt)
    x = task.get_input(idx)  # input
    ys = ['']  # current output candidates
    infos = []
    for step in range(task.steps):
        # generation
        if args.method_generate == 'sample':
            new_ys = [get_samples(task, x, y, args.n_generate_sample, prompt_sample=args.prompt_sample, stop=task.stops[step]) for y in ys]
        elif args.method_generate == 'propose':
            new_ys = [get_proposals(task, x, y) for y in ys]
        new_ys = list(itertools.chain(*new_ys))
        ids = list(range(len(new_ys)))
        # evaluation
        if args.method_evaluate == 'vote':
            values = get_votes(task, x, new_ys, args.n_evaluate_sample)
        elif args.method_evaluate == 'value':
            values = get_values(task, x, new_ys, args.n_evaluate_sample)

        # selection
        if args.method_select == 'sample':
            ps = np.array(values) / sum(values)
            select_ids = np.random.choice(ids, size=args.n_select_sample, p=ps).tolist()
        elif args.method_select == 'greedy':
            select_ids = sorted(ids, key=lambda x: values[x], reverse=True)[:args.n_select_sample]
        select_new_ys = [new_ys[select_id] for select_id in select_ids]

        # log
        if to_print: 
            sorted_new_ys, sorted_values = zip(*sorted(zip(new_ys, values), key=lambda x: x[1], reverse=True))
            print(f'-- new_ys --: {sorted_new_ys}\n-- sol values --: {sorted_values}\n-- choices --: {select_new_ys}\n')
        
        infos.append({'step': step, 'x': x, 'ys': ys, 'new_ys': new_ys, 'values': values, 'select_new_ys': select_new_ys})
        ys = select_new_ys

        should_stop = False
        for temp_ys in ys:
            if "24"  == get_current_numbers(temp_ys) or len(get_current_numbers(temp_ys).split(" ")) == 1:
                should_stop = True
                break
        if should_stop:
            break
        
    
    if to_print: 
        print(ys)
    correct_answer = None
    for temp_ys in ys:
        if "24"  == get_current_numbers(temp_ys):
            correct_answer = temp_ys
            break
    if correct_answer:
        print("found correct anser:", correct_answer)
    else:
        print("No correct answer found")

    return ys, {'steps': infos}

def naive_solve(args, task, idx, to_print=True):
    global gpt
    gpt = partial(gpt, model=args.backend, temperature=args.temperature)
    print(gpt)
    x = task.get_input(idx)  # input
    ys = get_samples(task, x, '', args.n_generate_sample, args.prompt_sample, stop=None)
    return ys, {}