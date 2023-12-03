import argparse
from tot.methods.bfs import solve
from tot.tasks.game24 import Game24Task
import time

args = argparse.Namespace(backend='gpt-4', temperature=0.7, task='game24', naive_run=False, prompt_sample=None, method_generate='propose', method_evaluate='value', method_select='greedy', n_generate_sample=1, n_evaluate_sample=3, n_select_sample=5)

t = time.localtime()
current_time = time.strftime("%H:%M:%S", t)
print(current_time)

# start 4 5 6 10
task = Game24Task()
ys, infos = solve(args, task, 900)
print(ys[0])

t = time.localtime()
current_time = time.strftime("%H:%M:%S", t)
print(current_time)
