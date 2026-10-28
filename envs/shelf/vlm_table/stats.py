import csv
import argparse

parser = argparse.ArgumentParser(description="Compute stats for a given cfg.")
parser.add_argument('--cfg', type=str, required=True, help='Model config name, e.g., gpt-5.2_LOW')
args = parser.parse_args()

NUM_RUNS = 20

cfg = args.cfg

n_prompts = 0
n_iter = 0
prompt_time = 0

n_runs = NUM_RUNS
n_success_runs = 0

n_total_problems = NUM_RUNS*4
n_success_problems = 0

max_prompts = 0
for i in range(NUM_RUNS):
    print(f"Processing {i}.csv")
    with open(f'envs/shelf/vlm_table/{cfg}/{i}.csv', mode='r', newline='', encoding='utf-8') as csvfile:
        # Create a DictReader object
        reader = csv.DictReader(csvfile)
        success = False
        
        # Iterate over each row
        for row in reader:
            if (row['success'] == 'True'):
                n_prompts += int(row['num_prompts'])
                n_iter += int(row['iterations'])
                prompt_time += float(row['prompting'])
                success = True
                n_success_problems += 1
                max_prompts = max(max_prompts, int(row['num_prompts']))

    if success:
        n_success_runs += 1

print(f"Stats for cfg {cfg}:")
print(f"THE FOLLOWING METRICS ARE AVERAGED OVER SUCCESSFUL RUNS ONLY=========================")
print(f"Number of successful runs: {n_success_runs}")
print(f"Total prompts: {n_prompts}")
print(f"Total iterations: {n_iter}")
print(f"Total prompting time: {prompt_time:.2f} seconds")
print("Avg per-prompt time: {:.2f} seconds".format(prompt_time / n_prompts if n_prompts > 0 else 0))
print("Avg refinement failures per run: {:.2f}".format(n_iter / n_success_runs if n_success_runs > 0 else 0))
print("Avg prompts per run: {:.2f}".format(n_prompts / n_success_runs if n_success_runs > 0 else 0))
print(f"Success rate: {n_success_problems}/{n_total_problems}")

print("Max prompts encountered during one problem instance: {:.2f}".format(max_prompts if max_prompts > 0 else 0))
