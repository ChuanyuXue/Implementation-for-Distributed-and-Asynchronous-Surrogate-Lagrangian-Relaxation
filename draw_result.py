import numpy as np
import copy
import time
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import pandas as pd
import docplex.mp.model as cpx

def load_dat(path): 
    '''
    return a dict of parameters.
    '''
    f = open(path, 'r')
    content = f.read()
    f.close()

    result = {}
    g = []
    for i in range(0, 20):
        g.append(content.split('\n')[13 + i])
    g[0] = g[0][5:]
    for index, gi in enumerate(g):
        g[index] = gi.replace('[','').replace(']','').split()
        g[index] = [int(x) for x in g[index]]
    result['objective'] = g
    a = []
    for i in range(0, 20):
        a.append(content.split('\n')[34 + i])
    a[0] = a[0][5:]
    for index, ai in enumerate(a):
        a[index] = ai.replace('[','').replace(']','').split()
        a[index] = [int(x) for x in a[index]]
    result['constraints'] = a
    b = content.split('\n')[56][2:].replace('[','').replace(']','').split()
    b = [int(x) for x in b]
    result['bs'] = b
    lambda_0 = content.split('\n')[60].replace('[','').replace(']','').replace(';','').split()
    lambda_0 = [-1 * int(x) for x in lambda_0]
    result['lambda_0'] = lambda_0
    return result

def calculate_feasible_solution(para, solutions):
    model = cpx.Model(name="MIP Model")
    model.time_limit = 180

    x_var = [model.binary_var_list(keys=list(range(1600)), name = 'x_%d'%i) for i in range(20)]
    z_var = model.binary_var_list(keys=list(range(20)), name = 'z')

    objective = model.sum([model.sum([para['objective'][i][k] * x_var[i][k] for k in range(1600)]) for i in range(20)])

    for j in range(1600):
        model.add_constraint(model.sum([x_var[i][j] for i in range(20)]) == 1)

    model.add_constraint(model.sum([z_var[i] for i in range(20)]) <= 12)

    for i in range(20):
        model.add_constraint(model.sum([para['constraints'][i][j] * x_var[i][j] for j in range(1600)]) <= para['bs'][i])
        for j in range(1600):
            model.add_constraint(z_var[i] - 1 <= 100*(x_var[i][j] - solutions[i][j] + 1))
            model.add_constraint(z_var[i] - 1 >= -100*(x_var[i][j] - solutions[i][j] + 1))
            model.add_constraint(z_var[i] - 1 <= -100*(x_var[i][j] - solutions[i][j] - 1))
            model.add_constraint(z_var[i] - 1 >= 100*(x_var[i][j] - solutions[i][j] - 1))
    
    model.minimize(objective)
    result = model.solve()
    return result


def calculate_dual_value(para, lambdas):
    result = []
    for i in range(20):
        model = cpx.Model(name="MIP Model")
        x_var = model.binary_var_list(keys=list(range(1600)), name = 'x_%d'%i) 

        objective = model.sum([(para['objective'][i][j] + lambdas[j]) * x_var[j] for j in range(1600)])

        model.add_constraint(model.sum([para['constraints'][i][j] * x_var[j] for j in range(1600)]) <= para['bs'][i])

        model.minimize(objective)
        result.append(model.solve().get_all_values())
    result = np.array(result)


    dual = sum([sum([para['objective'][i][j] * result[i][j] for i in range(20)]) for j in range(1600)]) + \
    sum([lambdas[j] * (
    sum([result[i][j] for i in range(20)]) - 1
    ) for j in range(1600)])
    
    return dual


para = load_dat('d201600.dat')

solutions = np.load('DASLRsolutions.npy')
lambdas = np.load('DASLRlambdas.npy')
costs = np.load('DASLRlogs.npy')

count = 0
logs = []
for index, row in enumerate(lambdas):
    count += 1
    if count % 500 == 0:
        time = row[0]
        re = calculate_feasible_solution(para, solutions[index][1:].reshape(20, 1600))
        feasible_cost = re.objective_value if re else -1
        re = calculate_dual_value(para, row[1:])
        dual_value = re
        cost = costs[index]
        print('Time %f ---- Feasible %f ---- Dual %f ---- Cost %f'%(time, feasible_cost, dual_value, cost[1]))
        logs.append([time, feasible_cost, dual_value, cost[1]])

solutions = np.load('SLRsolutions.npy')
lambdas = np.load('SLRlambdas.npy')
costs_slr = np.load('SLRlogs.npy')

count = 0
logs_slr = []
for index, row in enumerate(solutions):
    count += 1
    if count % 500 == 0:
        time = row[0]
        re = calculate_feasible_solution(para, row[1:].reshape(20, 1600))
        feasible_cost = re.objective_value if re else -1
        re = calculate_dual_value(para, lambdas[index][1:])
        dual_value = re
        print('Time %f ---- Feasible %f ---- Dual %f'%(time, feasible_cost, dual_value))
        logs_slr.append([time, feasible_cost, dual_value])

plt.figure(figsize=(20, 16))
plt.scatter([x[0] for x in logs], [x[1] for x in logs], label = 'DASLR feasible cost', marker='x')
plt.scatter([x[0] for x in logs], [x[2] for x in logs], label = 'DASLR dual value', color='steelblue')
plt.plot([x[0] for x in costs], [x[1] for x in costs], label = 'DASLR cost')

plt.scatter([x[0] for x in logs_slr], [x[1] for x in logs_slr], label = 'SLR feasible cost', marker='x')
plt.scatter([x[0] for x in logs_slr], [x[2] for x in logs_slr], label = 'SLR dual value', color = 'darkorange')
plt.plot([x[0] for x in costs_slr], [x[1] for x in costs_slr], label = 'SLR cost')

plt.ylim(97790, 97880)
plt.xlabel('Seconds')
plt.legend()
plt.save_fig('result_in_500sec.pdf')


plt.figure(figsize=(20, 16))
plt.plot([x[0] + 1 for x in costs], [x[2]**2 for x in costs], label = 'DASLR')
plt.plot([x[0] + 1 for x in costs_slr], [x[2]**2 for x in costs_slr], label = 'SLR')

plt.ylabel('Norm squred')
plt.xlabel('Seconds(log scaled)')
plt.xscale("log")
plt.legend()
plt.savefig('result_in_600sec_1.pdf')