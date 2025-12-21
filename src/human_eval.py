import pandas as pd
from pathlib import Path

# Use relative paths from project root
results_dir = Path(__file__).parent.parent / 'results' / 'mechanicalturk_csvs'
bach_vs_q_path = results_dir / 'bach_vs_q_batch_results.csv'
doodle_vs_q_path = results_dir / 'doodle_vs_q_batch_results.csv'
rand_vs_q_path = results_dir / 'rand_vs_q_batch_results.csv'

field = 'Answer.harmonization-preference.label'

bach_vs_q = pd.read_csv(bach_vs_q_path)
doodle_vs_q = pd.read_csv(doodle_vs_q_path)
rand_vs_q = pd.read_csv(rand_vs_q_path)

print("BACH")
res_series = bach_vs_q[field]
A_count = 0
B_count = 0
for i in range(250):
    res = res_series[i]
    if res == "Harmonization A":
        A_count +=1 
    else:
        B_count +=1

print("A COUNT:", A_count)
print("B COUNT:", B_count)


print("RANDOM")
res_series = rand_vs_q[field]
A_count = 0
B_count = 0
for i in range(250):
    res = res_series[i]
    if res == "Harmonization A":
        A_count +=1 
    else:
        B_count +=1

print("A COUNT:", A_count)
print("B COUNT:", B_count)


print("DOODLE")
res_series = doodle_vs_q[field]
A_count = 0
B_count = 0
for i in range(250):
    res = res_series[i]
    if res == "Harmonization A":
        A_count +=1 
    else:
        B_count +=1
print("A COUNT:", A_count)
print("B COUNT:", B_count)