
import pandas as pd
from utils import *

A = "results/results_covid/vicuna.txt"
B = "results/results_covid/vicuna_noda.txt"
C = "results/results_covid/zephyr.txt"
D = "results/results_covid/mistral.txt"


d_files = {'A': A, 'B': B, 'C': C, 'D': D}

elements = ['A', 'B', 'C', 'D']

all_combinations = generate_combinations(elements)


algorithms = ['RRF', 'Borda', 'My_method1', 'My_method2', 'AGG_RRF', 'AGG_Borda']
df_ndcg = pd.DataFrame(index=algorithms, columns=all_combinations)
num_comb = 0
for i in all_combinations:
    print("number of iteration: ", num_comb)
    num_comb += 1
    if len(i) == 2:
        app1 = d_files.get(i[0])
        app2 = d_files.get(i[1])
        with open(app1, 'r') as f1, open(app2, 'r') as f2:
            file1_lines = f1.readlines()
            file2_lines = f2.readlines()
        file1_numbers, _ = obtain_numbers(file1_lines)
        file2_numbers, _ = obtain_numbers(file2_lines)
        # apply_RRF_2(file1_numbers, file2_numbers, df_ndcg, i)
        # apply_borda_2(file1_numbers, file2_numbers, df_ndcg, i)
        apply_my_method1_2(file1_numbers, file2_numbers, df_ndcg, i, app1, app2)
        apply_my_method2_2(file1_numbers, file2_numbers, df_ndcg, i, app1, app2)
        apply_merging_merge_borda(df_ndcg, i)
        apply_merging_merge_RRF(df_ndcg, i)
    elif len(i) == 3:
        app1 = d_files.get(i[0])
        app2 = d_files.get(i[1])
        app3 = d_files.get(i[2])
        with open(app1, 'r') as f1, open(app2, 'r') as f2, open(app3, 'r') as f3:
            file1_lines = f1.readlines()
            file2_lines = f2.readlines()
            file3_lines = f3.readlines()
        file1_numbers, _ = obtain_numbers(file1_lines)
        file2_numbers, _ = obtain_numbers(file2_lines)
        file3_numbers, _ = obtain_numbers(file3_lines)
        apply_RRF_3(file1_numbers, file2_numbers, file3_numbers, df_ndcg, i)
        apply_borda_3(file1_numbers, file2_numbers, file3_numbers, df_ndcg, i)
        apply_my_method1_3(file1_numbers, file2_numbers, file3_numbers, df_ndcg, i, app1, app2, app3)
        apply_my_method2_3(file1_numbers, file2_numbers, file3_numbers, df_ndcg, i, app1, app2, app3)
        apply_merging_merge_borda(df_ndcg, i)
        apply_merging_merge_RRF(df_ndcg, i)
    elif len(i) == 4:
        app1 = d_files.get(i[0])
        app2 = d_files.get(i[1])
        app3 = d_files.get(i[2])
        app4 = d_files.get(i[3])
        with open(app1, 'r') as f1, open(app2, 'r') as f2, open(app3, 'r') as f3, open(app4, 'r') as f4:
            file1_lines = f1.readlines()
            file2_lines = f2.readlines()
            file3_lines = f3.readlines()
            file4_lines = f4.readlines()
        file1_numbers, _ = obtain_numbers(file1_lines)
        file2_numbers, _ = obtain_numbers(file2_lines)
        file3_numbers, _ = obtain_numbers(file3_lines)
        file4_numbers, _ = obtain_numbers(file4_lines)
        apply_RRF_4(file1_numbers, file2_numbers, file3_numbers, file4_numbers, df_ndcg, i)
        apply_borda_4(file1_numbers, file2_numbers, file3_numbers, file4_numbers, df_ndcg, i)
        apply_my_method1_4(file1_numbers, file2_numbers, file3_numbers, file4_numbers, df_ndcg, i, app1, app2, app3, app4)
        apply_my_method2_4(file1_numbers, file2_numbers, file3_numbers, file4_numbers, df_ndcg, i, app1, app2, app3, app4)
        apply_merging_merge_borda(df_ndcg, i)
        apply_merging_merge_RRF(df_ndcg, i)
# save df to csv
df_ndcg.to_csv('results/results_covid/df_ndcg20.csv')