import math
import os

from itertools import combinations


import subprocess
def run_command():
    command = [
        'python', 
        'rank_llm/src/rank_llm/scripts/run_rank_llm.py',
        '--model_path=castorini/rank_vicuna_7b_v1',
        '--top_k_candidates=20',
        '--dataset=covid',
        '--retrieval_method=SPLADE++_EnsembleDistil_ONNX',
        '--prompt_mode=rank_GPT',
        '--context_size=4096',
        '--variable_passages'
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    return float(result.stdout.split('\n')[-2].split('\t')[-1])




def generate_combinations(elements):
    all_combinations = []
    for r in range(2, len(elements) + 1): 
        combinations_r = list(combinations(elements, r))
        all_combinations.extend(combinations_r)
    return all_combinations

def obtain_only_numbers(lines):
    file_numbers = []
    for i, line in enumerate(lines):
        file_numbers.append([int(num) for num in line.strip().split(',')])
    return file_numbers


def obtain_numbers(lines):
    file_numbers = []
    file_probabilities = []
    for i, line in enumerate(lines):
        if i % 2 == 0:  # Even lines, for numbers
            file_numbers.append([int(num) for num in line.strip().split(',')])
        else:  # Odd lines, for probabilities
            file_probabilities.append([float(prob) for prob in line.strip().split(',')])
    return file_numbers, file_probabilities

################ Methods ################


def linear(index):
    return -0.04*index+1.08

def logistic(index):
    return  1/math.log2((index+2))

def regression_function_zephyr(x):
    return -0.00423579 * x + 0.8144797814207652


def regression_function_vicuna(x):
    return -0.00528333 * x + 0.7872333333333333

# Define the regression function
def regression_function_vicuna_noda(x):
    return -0.00565656 * x + 0.7930683060109288

def regression_function_mistral(x):
    return -0.00645769 * x + 0.8203128205128206

function_map = {
    'regression_function_vicuna': regression_function_vicuna,
    'regression_function_vicuna_noda': regression_function_vicuna_noda,
    'regression_function_zephyr': regression_function_zephyr,
    'regression_function_mistral': regression_function_mistral
}



def RRF_2(index1, index2, k=60):
    return 1/(k+index1+1) + 1/(k+index2+1) 

def RRF_3(index1, index2, index3, k=60):
    return 1/(k+index1+1) + 1/(k+index2+1) + 1/(k+index3+1)

def RRF_4(index1, index2, index3, index4, k=60):
    return 1/(k+index1+1) + 1/(k+index2+1) + 1/(k+index3+1) + 1/(k+index4+1)

def RRF_5(index1, index2, index3, index4, index5, k=60):
    return 1/(k+index1+1) + 1/(k+index2+1) + 1/(k+index3+1) + 1/(k+index4+1) + 1/(k+index5+1)

def borda_2(index1, index2):
    cnt = 20
    sum1 = (cnt-index1)
    sum2 = (cnt-index2)
    return sum1 + sum2

def borda_3(index1, index2, index3):
    cnt = 20
    sum1 = (cnt-index1)
    sum2 = (cnt-index2)
    sum3 = (cnt-index3)
    return sum1 + sum2 + sum3

def borda_4(index1, index2, index3, index4):
    cnt = 20
    sum1 = (cnt-index1)
    sum2 = (cnt-index2)
    sum3 = (cnt-index3)
    sum4 = (cnt-index4)
    return sum1 + sum2 + sum3 + sum4

def borda_5(index1, index2, index3, index4, index5):
    cnt = 20
    sum1 = (cnt-index1)
    sum2 = (cnt-index2)
    sum3 = (cnt-index3)
    sum4 = (cnt-index4)
    sum5 = (cnt-index5)
    return sum1 + sum2 + sum3 + sum4 + sum5


def my_method1_2(index1, index2, app1, app2):
    cnt = 20
    name1 = app1.split('.txt')[0].split('/')[-1]
    func1 = function_map.get('regression_function_' + name1)
    name_2 = app2.split('.txt')[0].split('/')[-1]
    func2 = function_map.get('regression_function_' + name_2)
    sum1 = (cnt-index1)  * linear(index1) * func1(index1)
    sum2 = (cnt-index2)  * linear(index2) * func2(index2)
    return sum1 + sum2

def my_method1_3(index1, index2, index3, app1, app2, app3):
    cnt = 20
    name1 = app1.split('.txt')[0].split('/')[-1]
    func1 = function_map.get('regression_function_' + name1)
    name_2 = app2.split('.txt')[0].split('/')[-1]
    func2 = function_map.get('regression_function_' + name_2)
    name_3 = app3.split('.txt')[0].split('/')[-1]
    func3 = function_map.get('regression_function_' + name_3)
    sum1 = (cnt-index1)  * linear(index1) *  func1(index1)
    sum2 = (cnt-index2)  * linear(index2) * func2(index2) 
    sum3 = (cnt-index3)  * linear(index3) * func3(index3)
    return sum1 + sum2 + sum3

def my_method1_4(index1, index2, index3, index4, app1, app2, app3, app4):
    cnt = 20
    name1 = app1.split('.txt')[0].split('/')[-1]
    func1 = function_map.get('regression_function_' + name1)
    name_2 = app2.split('.txt')[0].split('/')[-1]
    func2 = function_map.get('regression_function_' + name_2)
    name_3 = app3.split('.txt')[0].split('/')[-1]
    func3 = function_map.get('regression_function_' + name_3)
    name_4 = app4.split('.txt')[0].split('/')[-1]
    func4 = function_map.get('regression_function_' + name_4)
    sum1 = (cnt-index1)  * linear(index1) * func1(index1)
    sum2 = (cnt-index2)  * linear(index2) * func2(index2) 
    sum3 = (cnt-index3)  * linear(index3) * func3(index3)
    sum4 = (cnt-index4)  * linear(index4) * func4(index4)
    return sum1 + sum2 + sum3 + sum4

def my_method2_2(index1, index2, app1, app2):
    cnt = 20
    name1 = app1.split('.txt')[0].split('/')[-1]
    func1 = function_map.get('regression_function_' + name1)
    name_2 = app2.split('.txt')[0].split('/')[-1]
    func2 = function_map.get('regression_function_' + name_2)
    sum1 = (cnt-index1) * logistic(index1) * func1(index1)
    sum2 = (cnt-index2) * logistic(index2) * func2(index2)
    return sum1 + sum2

def my_method2_3(index1, index2, index3, app1, app2, app3):
    cnt = 20
    name1 = app1.split('.txt')[0].split('/')[-1]
    func1 = function_map.get('regression_function_' + name1)
    name_2 = app2.split('.txt')[0].split('/')[-1]
    func2 = function_map.get('regression_function_' + name_2)
    name_3 = app3.split('.txt')[0].split('/')[-1]
    func3 = function_map.get('regression_function_' + name_3)
    sum1 = (cnt-index1) * logistic(index1) * func1(index1)
    sum2 = (cnt-index2) * logistic(index2) * func2(index2)
    sum3 = (cnt-index3) * logistic(index3) * func3(index3)
    return sum1 + sum2 + sum3


def my_method2_4(index1, index2, index3, index4, app1, app2, app3, app4):
    cnt = 20
    name1 = app1.split('.txt')[0].split('/')[-1]
    func1 = function_map.get('regression_function_' + name1)
    name_2 = app2.split('.txt')[0].split('/')[-1]
    func2 = function_map.get('regression_function_' + name_2)
    name_3 = app3.split('.txt')[0].split('/')[-1]
    func3 = function_map.get('regression_function_' + name_3)
    name_4 = app4.split('.txt')[0].split('/')[-1]
    func4 = function_map.get('regression_function_' + name_4)
    sum1 = (cnt-index1) * logistic(index1) * func1(index1)
    sum2 = (cnt-index2) * logistic(index2) * func2(index2)
    sum3 = (cnt-index3) * logistic(index3) * func3(index3)
    sum4 = (cnt-index4) * logistic(index4) * func4(index4)
    return sum1 + sum2 + sum3 + sum4


################################## APPLY METHODS ##################################



def apply_RRF_2(file1_numbers, file2_numbers, df_ndcg10, combination):
    os.remove('results/best_two_models.txt')
    os.remove('results/RRF.txt')
    best = []
    for i in range(len(file1_numbers)):
        d = {}
        for k in range(len(file1_numbers[i])):
            index1 = file1_numbers[i].index(file1_numbers[i][k])  
            index2 = file2_numbers[i].index(file1_numbers[i][k])  
            s = RRF_2(index1, index2)
            d[file1_numbers[i][k]] = s
        best.append(d)

    with open("results/best_two_models.txt", "a") as output_file:
        for j in best:
            sorted_dict = dict(sorted(j.items(), key=lambda item: item[1], reverse=True))
            for i, num in enumerate(sorted_dict):
                output_file.write(str(num))
                if i < len(j) - 1:
                    output_file.write(",")
            output_file.write('\n')
    with open("results/RRF.txt", "a") as output_file:
        for j in best:
            sorted_dict = dict(sorted(j.items(), key=lambda item: item[1], reverse=True))
            for i, num in enumerate(sorted_dict):
                output_file.write(str(num))
                if i < len(j) - 1:
                    output_file.write(",")
            output_file.write('\n')
    A = run_command()
    df_ndcg10.at['RRF', combination] = A


def apply_RRF_3(file1_numbers, file2_numbers, file3_numbers, df_ndcg10, combination):
    os.remove('results/best_two_models.txt')
    os.remove('results/RRF.txt')
    best = []
    for i in range(len(file1_numbers)):
        d = {}
        for k in range(len(file1_numbers[i])):
            index1 = file1_numbers[i].index(file1_numbers[i][k])  
            index2 = file2_numbers[i].index(file1_numbers[i][k])  
            index3 = file3_numbers[i].index(file1_numbers[i][k])  
            s = RRF_3(index1, index2, index3)
            d[file1_numbers[i][k]] = s
        best.append(d)

    with open("results/best_two_models.txt", "a") as output_file:
        for j in best:
            sorted_dict = dict(sorted(j.items(), key=lambda item: item[1], reverse=True))
            for i, num in enumerate(sorted_dict):
                output_file.write(str(num))
                if i < len(j) - 1:
                    output_file.write(",")
            output_file.write('\n')
    with open("results/RRF.txt", "a") as output_file:
        for j in best:
            sorted_dict = dict(sorted(j.items(), key=lambda item: item[1], reverse=True))
            for i, num in enumerate(sorted_dict):
                output_file.write(str(num))
                if i < len(j) - 1:
                    output_file.write(",")
            output_file.write('\n')
    A = run_command()
    df_ndcg10.at['RRF', combination] = A

def apply_RRF_4(file1_numbers, file2_numbers, file3_numbers, file4_numbers, df_ndcg10, combination):
    os.remove('results/best_two_models.txt')
    os.remove('results/RRF.txt')
    best = []
    for i in range(len(file1_numbers)):
        d = {}
        for k in range(len(file1_numbers[i])):
            index1 = file1_numbers[i].index(file1_numbers[i][k])  
            index2 = file2_numbers[i].index(file1_numbers[i][k])  
            index3 = file3_numbers[i].index(file1_numbers[i][k]) 
            index4 = file4_numbers[i].index(file1_numbers[i][k]) 
            s = RRF_4(index1, index2, index3, index4)
            d[file1_numbers[i][k]] = s
        best.append(d)

    with open("results/best_two_models.txt", "a") as output_file:
        for j in best:
            sorted_dict = dict(sorted(j.items(), key=lambda item: item[1], reverse=True))
            for i, num in enumerate(sorted_dict):
                output_file.write(str(num))
                if i < len(j) - 1:
                    output_file.write(",")
            output_file.write('\n')
    with open("results/RRF.txt", "a") as output_file:
        for j in best:
            sorted_dict = dict(sorted(j.items(), key=lambda item: item[1], reverse=True))
            for i, num in enumerate(sorted_dict):
                output_file.write(str(num))
                if i < len(j) - 1:
                    output_file.write(",")
            output_file.write('\n')
    A = run_command()
    df_ndcg10.at['RRF', combination] = A



def apply_borda_2(file1_numbers, file2_numbers, df_ndcg10, combination):
    os.remove('results/best_two_models.txt')
    os.remove('results/Borda.txt')
    best = []
    for i in range(len(file1_numbers)):
        d = {}
        for k in range(len(file1_numbers[i])):
            index1 = file1_numbers[i].index(file1_numbers[i][k])  
            index2 = file2_numbers[i].index(file1_numbers[i][k])  
            s = borda_2(index1, index2)
            d[file1_numbers[i][k]] = s
        best.append(d)

    with open("results/best_two_models.txt", "a") as output_file:
        for j in best:
            sorted_dict = dict(sorted(j.items(), key=lambda item: item[1], reverse=True))
            for i, num in enumerate(sorted_dict):
                output_file.write(str(num))
                if i < len(j) - 1:
                    output_file.write(",")
            output_file.write('\n')
    with open("results/Borda.txt", "a") as output_file:
        for j in best:
            sorted_dict = dict(sorted(j.items(), key=lambda item: item[1], reverse=True))
            for i, num in enumerate(sorted_dict):
                output_file.write(str(num))
                if i < len(j) - 1:
                    output_file.write(",")
            output_file.write('\n')
    A = run_command()
    df_ndcg10.at['Borda', combination] = A

def apply_borda_3(file1_numbers, file2_numbers, file3_numbers, df_ndcg10, combination):
    os.remove('results/best_two_models.txt')
    os.remove('results/Borda.txt')
    best = []
    for i in range(len(file1_numbers)):
        d = {}
        for k in range(len(file1_numbers[i])):
            index1 = file1_numbers[i].index(file1_numbers[i][k])  
            index2 = file2_numbers[i].index(file1_numbers[i][k])  
            index3 = file3_numbers[i].index(file1_numbers[i][k]) 
            s = borda_3(index1, index2, index3)
            d[file1_numbers[i][k]] = s
        best.append(d)

    with open("results/best_two_models.txt", "a") as output_file:
        for j in best:
            sorted_dict = dict(sorted(j.items(), key=lambda item: item[1], reverse=True))
            for i, num in enumerate(sorted_dict):
                output_file.write(str(num))
                if i < len(j) - 1:
                    output_file.write(",")
            output_file.write('\n')
    with open("results/Borda.txt", "a") as output_file:
        for j in best:
            sorted_dict = dict(sorted(j.items(), key=lambda item: item[1], reverse=True))
            for i, num in enumerate(sorted_dict):
                output_file.write(str(num))
                if i < len(j) - 1:
                    output_file.write(",")
            output_file.write('\n')
    A = run_command()
    df_ndcg10.at['Borda', combination] = A


def apply_borda_4(file1_numbers, file2_numbers, file3_numbers, file4_numbers, df_ndcg10, combination):
    os.remove('results/best_two_models.txt')
    os.remove('results/Borda.txt')
    best = []
    for i in range(len(file1_numbers)):
        d = {}
        for k in range(len(file1_numbers[i])):
            index1 = file1_numbers[i].index(file1_numbers[i][k])  
            index2 = file2_numbers[i].index(file1_numbers[i][k])  
            index3 = file3_numbers[i].index(file1_numbers[i][k])  
            index4 = file4_numbers[i].index(file1_numbers[i][k])  
            s = borda_4(index1, index2, index3, index4)
            d[file1_numbers[i][k]] = s
        best.append(d)

    with open("results/best_two_models.txt", "a") as output_file:
        for j in best:
            sorted_dict = dict(sorted(j.items(), key=lambda item: item[1], reverse=True))
            for i, num in enumerate(sorted_dict):
                output_file.write(str(num))
                if i < len(j) - 1:
                    output_file.write(",")
            output_file.write('\n')
    with open("results/Borda.txt", "a") as output_file:
        for j in best:
            sorted_dict = dict(sorted(j.items(), key=lambda item: item[1], reverse=True))
            for i, num in enumerate(sorted_dict):
                output_file.write(str(num))
                if i < len(j) - 1:
                    output_file.write(",")
            output_file.write('\n')
    A = run_command()
    df_ndcg10.at['Borda', combination] = A



def apply_my_method1_2(file1_numbers, file2_numbers, df_ndcg10, combination, app1, app2):
    os.remove('results/best_two_models.txt')
    os.remove('results/My_1.txt')
    best = []
    for i in range(len(file1_numbers)):
        d = {}
        for k in range(len(file1_numbers[i])):
            index1 = file1_numbers[i].index(file1_numbers[i][k])  
            index2 = file2_numbers[i].index(file1_numbers[i][k])  
            s = my_method1_2(index1, index2, app1, app2)
            d[file1_numbers[i][k]] = s
        best.append(d)

    with open("results/best_two_models.txt", "a") as output_file:
        for j in best:
            sorted_dict = dict(sorted(j.items(), key=lambda item: item[1], reverse=True))
            for i, num in enumerate(sorted_dict):
                output_file.write(str(num))
                if i < len(j) - 1:
                    output_file.write(",")
            output_file.write('\n')
    with open("results/My_1.txt", "a") as output_file:
        for j in best:
            sorted_dict = dict(sorted(j.items(), key=lambda item: item[1], reverse=True))
            for i, num in enumerate(sorted_dict):
                output_file.write(str(num))
                if i < len(j) - 1:
                    output_file.write(",")
            output_file.write('\n')
    A = run_command()
    df_ndcg10.at['My_method1', combination] = A

def apply_my_method1_3(file1_numbers, file2_numbers, file3_numbers, df_ndcg10, combination, app1, app2, app3):
    os.remove('results/best_two_models.txt')
    os.remove('results/My_1.txt')
    best = []
    for i in range(len(file1_numbers)):
        d = {}
        for k in range(len(file1_numbers[i])):
            index1 = file1_numbers[i].index(file1_numbers[i][k]) 
            index2 = file2_numbers[i].index(file1_numbers[i][k])  
            index3 = file3_numbers[i].index(file1_numbers[i][k])  
            s = my_method1_3(index1, index2, index3, app1, app2, app3)
            d[file1_numbers[i][k]] = s
        best.append(d)

    with open("results/best_two_models.txt", "a") as output_file:
        for j in best:
            sorted_dict = dict(sorted(j.items(), key=lambda item: item[1], reverse=True))
            for i, num in enumerate(sorted_dict):
                output_file.write(str(num))
                if i < len(j) - 1:
                    output_file.write(",")
            output_file.write('\n')
    with open("results/My_1.txt", "a") as output_file:
        for j in best:
            sorted_dict = dict(sorted(j.items(), key=lambda item: item[1], reverse=True))
            for i, num in enumerate(sorted_dict):
                output_file.write(str(num))
                if i < len(j) - 1:
                    output_file.write(",")
            output_file.write('\n')
    A = run_command()
    df_ndcg10.at['My_method1', combination] = A


def apply_my_method1_4(file1_numbers, file2_numbers, file3_numbers, file4_numbers, df_ndcg10, combination, app1, app2, app3, app4):
    os.remove('results/best_two_models.txt')
    os.remove('results/My_1.txt')
    best = []
    for i in range(len(file1_numbers)):
        d = {}
        for k in range(len(file1_numbers[i])):
            index1 = file1_numbers[i].index(file1_numbers[i][k])  
            index2 = file2_numbers[i].index(file1_numbers[i][k])  
            index3 = file3_numbers[i].index(file1_numbers[i][k])  
            index4 = file4_numbers[i].index(file1_numbers[i][k])  
            s = my_method1_4(index1, index2, index3, index4, app1, app2, app3, app4)
            d[file1_numbers[i][k]] = s
        best.append(d)

    with open("results/best_two_models.txt", "a") as output_file:
        for j in best:
            sorted_dict = dict(sorted(j.items(), key=lambda item: item[1], reverse=True))
            for i, num in enumerate(sorted_dict):
                output_file.write(str(num))
                if i < len(j) - 1:
                    output_file.write(",")
            output_file.write('\n')
    with open("results/My_1.txt", "a") as output_file:
        for j in best:
            sorted_dict = dict(sorted(j.items(), key=lambda item: item[1], reverse=True))
            for i, num in enumerate(sorted_dict):
                output_file.write(str(num))
                if i < len(j) - 1:
                    output_file.write(",")
            output_file.write('\n')

    A = run_command()
    df_ndcg10.at['My_method1', combination] = A



def apply_my_method2_2(file1_numbers, file2_numbers, df_ndcg10, combination, app1, app2):
    os.remove('results/best_two_models.txt')
    os.remove('results/My_2.txt')
    best = []
    for i in range(len(file1_numbers)):
        d = {}
        for k in range(len(file1_numbers[i])):
            index1 = file1_numbers[i].index(file1_numbers[i][k])  
            index2 = file2_numbers[i].index(file1_numbers[i][k])  
            s = my_method2_2(index1, index2, app1, app2)
            d[file1_numbers[i][k]] = s
        best.append(d)

    with open("results/best_two_models.txt", "a") as output_file:
        for j in best:
            sorted_dict = dict(sorted(j.items(), key=lambda item: item[1], reverse=True))
            for i, num in enumerate(sorted_dict):
                output_file.write(str(num))
                if i < len(j) - 1:
                    output_file.write(",")
            output_file.write('\n')
    with open("results/My_2.txt", "a") as output_file:
        for j in best:
            sorted_dict = dict(sorted(j.items(), key=lambda item: item[1], reverse=True))
            for i, num in enumerate(sorted_dict):
                output_file.write(str(num))
                if i < len(j) - 1:
                    output_file.write(",")
            output_file.write('\n')
    A = run_command()
    df_ndcg10.at['My_method2', combination] = A

def apply_my_method2_3(file1_numbers, file2_numbers, file3_numbers, df_ndcg10, combination, app1, app2, app3):
    os.remove('results/best_two_models.txt')
    os.remove('results/My_2.txt')
    best = []
    for i in range(len(file1_numbers)):
        d = {}
        for k in range(len(file1_numbers[i])):
            index1 = file1_numbers[i].index(file1_numbers[i][k])  
            index2 = file2_numbers[i].index(file1_numbers[i][k])  
            index3 = file3_numbers[i].index(file1_numbers[i][k])  
            s = my_method2_3(index1, index2, index3, app1, app2, app3)
            d[file1_numbers[i][k]] = s
        best.append(d)

    with open("results/best_two_models.txt", "a") as output_file:
        for j in best:
            sorted_dict = dict(sorted(j.items(), key=lambda item: item[1], reverse=True))
            for i, num in enumerate(sorted_dict):
                output_file.write(str(num))
                if i < len(j) - 1:
                    output_file.write(",")
            output_file.write('\n')

    with open("results/My_2.txt", "a") as output_file:
        for j in best:
            sorted_dict = dict(sorted(j.items(), key=lambda item: item[1], reverse=True))
            for i, num in enumerate(sorted_dict):
                output_file.write(str(num))
                if i < len(j) - 1:
                    output_file.write(",")
            output_file.write('\n')
    
    A = run_command()
    df_ndcg10.at['My_method2', combination] = A

def apply_my_method2_4(file1_numbers, file2_numbers, file3_numbers, file4_numbers, df_ndcg10, combination, app1, app2, app3, app4):
    os.remove('results/best_two_models.txt')
    os.remove('results/My_2.txt')
    best = []
    for i in range(len(file1_numbers)):
        d = {}
        for k in range(len(file1_numbers[i])):
            index1 = file1_numbers[i].index(file1_numbers[i][k]) 
            index2 = file2_numbers[i].index(file1_numbers[i][k])  
            index3 = file3_numbers[i].index(file1_numbers[i][k])  
            index4 = file4_numbers[i].index(file1_numbers[i][k])  
            s = my_method2_4(index1, index2, index3, index4, app1, app2, app3, app4)
            d[file1_numbers[i][k]] = s
        best.append(d)

    with open("results/best_two_models.txt", "a") as output_file:
        for j in best:
            sorted_dict = dict(sorted(j.items(), key=lambda item: item[1], reverse=True))
            for i, num in enumerate(sorted_dict):
                output_file.write(str(num))
                if i < len(j) - 1:
                    output_file.write(",")
            output_file.write('\n')
    with open("results/My_2.txt", "a") as output_file:
        for j in best:
            sorted_dict = dict(sorted(j.items(), key=lambda item: item[1], reverse=True))
            for i, num in enumerate(sorted_dict):
                output_file.write(str(num))
                if i < len(j) - 1:
                    output_file.write(",")
            output_file.write('\n')
    A = run_command()
    df_ndcg10.at['My_method2', combination] = A



def apply_merging_merge_RRF(df_ndcg10, combination):
    os.remove('results/best_two_models.txt')
    with open('results/RRF.txt', 'r') as f1, open('results/Borda.txt', 'r') as f2, open('results/My_1.txt', 'r') as f3, open('results/My_2.txt', 'r') as f4:
        lines1 = f1.readlines()
        lines2 = f2.readlines()
        lines3 = f3.readlines()
        lines4 = f4.readlines()
    file1_numbers = obtain_only_numbers(lines1)
    file2_numbers = obtain_only_numbers(lines2)
    file3_numbers = obtain_only_numbers(lines3)
    file4_numbers = obtain_only_numbers(lines4)
    best = []

    for i in range(len(file1_numbers)):
            d = {}
            for k in range(len(file1_numbers[i])):
                index1 = file1_numbers[i].index(file1_numbers[i][k]) 
                index2 = file2_numbers[i].index(file1_numbers[i][k])  
                index3 = file3_numbers[i].index(file1_numbers[i][k])  
                index4 = file4_numbers[i].index(file1_numbers[i][k])  
                s = RRF_4(index1, index2, index3, index4)
                d[file1_numbers[i][k]] = s
            best.append(d)

    with open("results/best_two_models.txt", "a") as output_file:
        for j in best:
            sorted_dict = dict(sorted(j.items(), key=lambda item: item[1], reverse=True))
            for i, num in enumerate(sorted_dict):
                output_file.write(str(num))
                if i < len(j) - 1:
                    output_file.write(",")
            output_file.write('\n')

    A = run_command()
    df_ndcg10.at['AGG_RRF', combination] = A

def apply_merging_merge_borda(df_ndcg10, combination):
    os.remove('results/best_two_models.txt')
    with open('results/RRF.txt', 'r') as f1, open('results/Borda.txt', 'r') as f2, open('results/My_1.txt', 'r') as f3, open('results/My_2.txt', 'r') as f4:
        lines1 = f1.readlines()
        lines2 = f2.readlines()
        lines3 = f3.readlines()
        lines4 = f4.readlines()
    file1_numbers = obtain_only_numbers(lines1)
    file2_numbers = obtain_only_numbers(lines2)
    file3_numbers = obtain_only_numbers(lines3)
    file4_numbers = obtain_only_numbers(lines4)
    best = []

    for i in range(len(file1_numbers)):
            d = {}
            for k in range(len(file1_numbers[i])):
                index1 = file1_numbers[i].index(file1_numbers[i][k]) 
                index2 = file2_numbers[i].index(file1_numbers[i][k])  
                index3 = file3_numbers[i].index(file1_numbers[i][k])  
                index4 = file4_numbers[i].index(file1_numbers[i][k])  
                s = borda_4(index1, index2, index3, index4)
                d[file1_numbers[i][k]] = s
            best.append(d)

    with open("results/best_two_models.txt", "a") as output_file:
        for j in best:
            sorted_dict = dict(sorted(j.items(), key=lambda item: item[1], reverse=True))
            for i, num in enumerate(sorted_dict):
                output_file.write(str(num))
                if i < len(j) - 1:
                    output_file.write(",")
            output_file.write('\n')

    A = run_command()
    df_ndcg10.at['AGG_Borda', combination] = A
