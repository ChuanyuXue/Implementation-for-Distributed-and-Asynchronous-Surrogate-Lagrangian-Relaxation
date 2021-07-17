import numpy as np
import copy
import time
import docplex.mp.model as cpx
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm

class Coordinator(object):
    '''
    Coordinator for updating and broadcasting multipliers.
    '''
    def __init__(self, M = 75, r = 0.05, L_star = 97887, time = 100, threshold = 0.1):
        super().__init__()
        self._M = M
        self._r = r
        self._L_star = L_star
        self.time = time
        self.threshold = threshold
        
        
        self._k = None
        self._g = None
        self._s = None
        self._lambda = None
        self._solutions = None
        self._feasible_solutions = None
        self._feasible_cost = None
        self._surrogate_dual = None
        self._lower_bound = None
        self._dual_gap = None
        self._subproblems = None
        

        
    def init(self, para, subproblems):
        '''
        Initialize parameters, multipliers and stepsize, all subproblems are required to be initialized 
        and started in advance.
        
        para: Dict to describe the MILP problem
                    {'objective': [a1,a2,...,aI], 
                    'constraints': [[c11,c12,...,c1I],[c21,c22,...,c2I],...,[cm1,cm2,...cmI]]
                    'bs': [b1,b2,....,bm]}
                    
        subproblems: Iteriable predifined subproblems [sub1, sub2, .... , subI]
        
        '''
        self.objective = para['objective']
        self.constraints = para['constraints']
        self.bs = para['bs']
        self._subproblems = subproblems
        self._solutions = [None for x in range(len(self._subproblems))]
        
        if 'lambda_0' not in para:
            self._k, self._lambda = self._init_lambda()
        else:
            self._k, self._lambda = (-1, para['lambda_0'])
        for index, sub in enumerate(self._subproblems):
            self._solutions[index] = sub.run((self._k, self._lambda))
        
        ## Reset the paprameters
        self._k = 0
                
        ## Update subgradient with x^0
        self._g = np.array(
            [np.sum([self._solutions[x][i] for x in range(len(self._subproblems))]) - 1 for i in range(len(self._lambda))]
        )
        ## Update surrogate_dual with x^0 and g^0
        self._surrogate_dual = self._get_surrogate_dual()
        ## Update s^0 with x^0, surrogate_dual and g^0
        self._s = (1 / 40) * (self._L_star - self._surrogate_dual) / np.linalg.norm(self._g)**2
#         self._s = 0.005
        
    def run(self):
        '''
        Start and keep updating multipliers.
        '''
        self.times = []
        self.logs = []
        self.solutions = []
        self.lambdas = []
        
        start_time = time.time()
        ## Ignoring the stop creteria for debug
        self._dual_gap = 99999
        ## Start from k = 1
        current_time_diff = 0
        sequence = np.random.randint(0, len(self._subproblems), 1000000)
        while current_time_diff <= self.time and self._dual_gap >= self.threshold:
            ## Update lambda
            index, x = sequence[self._k], self._subproblems[sequence[self._k]].run((self._k, self._lambda))
            self._old_solution = self._solutions[index]
            self._solutions[index] = x
            self._old_g = self._g
            self._g = self._get_g(index)
            self._k = self._k + 1
            self._s = self._get_s()
            self._surrogate_dual = self._get_surrogate_dual()
            self._lambda = self._lambda + self._s * self._g
            current_time_diff = time.time() - start_time
    
            self.times.append(current_time_diff)
            self.logs.append([self._surrogate_dual, np.linalg.norm(self._g)])
            self.solutions.append([x for y in copy.deepcopy(self._solutions) for x in y])
            self.lambdas.append(copy.deepcopy(self._lambda))

        end_time = time.time() - start_time
        
        
    def _init_lambda(self):
        return (-1, np.zeros(shape = len(self.constraints[0])) + 0.1)
    
    def _get_g(self, index):
        '''
        This method can only use after updating x
        '''
        return np.array([self._g[x] - self._old_solution[x] + self._solutions[index][x] for x in range(len(self._lambda))])
    
    
    def _get_s(self):
        if self._k == 0:
            return self._s
        p = 1 - (1 / (self._k ** self._r))
        alpha = 1 - (1 / (self._M * (self._k ** p)))
        s_new = alpha * self._s * (np.linalg.norm(self._old_g, ord=2) / np.linalg.norm(self._g, ord=2))
        return s_new
    
    def _project_lambda(self):
        for i,x in enumerate(self._lambda):
            if x < 0:
                self._lambda[i] = 0
    
    def _get_surrogate_dual(self):
        '''
        This method can only use after updating x and g!
        '''
        return np.sum([(np.array(self._solutions[x])).dot(np.array(self.objective[x]) + np.array(self._lambda))
                       for x in range(len(self._subproblems))]) - np.sum(self._lambda)
    

class Solver():
    '''
    Solver for solving subproblems and sending them back to coordinator.
    
    index: int value indicates the number of subproblem.
    '''
    def __init__(self, index):
        self._lambda = None
        self._x = None
        self._k = 0
        
        self._lambda_index = -9999
        
        self._index = index
        self._name = 'x%d'%self._index
        
    def init(self, para, coordinator):
        '''
        para: Dict to describe the subproblem
              objective_g is generated from the objective function of original problem.
              objective_c is generated from the constraints of original problem.
                    {'objective_g': [g1,g2,...,gI],
                    'objective_c': [oc1,oc2,...,ocI],
                    'constraints': [[c11,c12,...,c1I],[c21,c22,...,c2I],...,[cm1,cm2,...cmI]]
                    'bs': [b1,b2,....,bm]}
        
        coordinator: Predifined coordinator
        '''
        self._k = -1
        self.objective_g = para['objective_g']
        self.objective_c = para['objective_c']
        self.constraints = para['constraints']
        self.bs = para['bs'] 
        self.ub = para['ub']
        self.lb = para['lb']
        
    def run(self, message):
        if message[0] > self._lambda_index:
            self._lambda = message[1]
            self._lambda_index = message[0]

            ## Solve solution with latest lambda
            self.solve()
            self._k = self._k + 1

            ## simulating the time delay of 5g transmission
            time.sleep(0.001)
            return self._x
        
                
    def solve(self,):
        model = cpx.Model(name="MIP Model")
        model.context.cplex_parameters.threads = 1
        x_var = model.binary_var_list(keys=list(range(len(self._lambda))), lb = self.lb, ub = self.ub, name = self._name)
        objective = model.sum([(self._lambda[i] * self.objective_c[i] + self.objective_g[i]) * x_var[i] for i in range(len(self._lambda))])
        model.add_constraint(model.sum([self.constraints[i] * x_var[i] for i in range(len(x_var))]) <= self.bs)
        model.minimize(objective)
        result = model.solve()
        self._x = result.get_all_values()
        
        
    def __str__(self):
        return self._name
    
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
    lambda_0 = [int(x) for x in lambda_0]
    result['lambda_0'] = lambda_0
    return result

para = load_dat('d201600.dat')

sub_paras = []
for j in range(20):
    sub_para = {
        'objective_g': para['objective'][j],
        'objective_c': [1 for i in range(len(para['objective'][j]))],
        'constraints': para['constraints'][j],
        'bs': para['bs'][j],
        'ub': 1,
        'lb': 0
    }
    sub_paras.append(sub_para)


## Fix the granularity of time stamp to coordinate with SLR

def fix_timestamp(logs):
    fix = list(logs)

    fix_x = [x[0] // 0.1 for x in fix]
    fix_y = [x[1] for x in fix]
    fix_z = [x[2] for x in fix]

    fix = dict()

    for i in range(len(fix_x)):
        fix[fix_x[i]] = [fix_y[i], fix_z[i]]

    logs = []
    for key in fix:
        logs.append([key / 10, fix[key][0], fix[key][1]])
    return logs

def save_result(coordinator, path = './', header=''):
    logs = np.array(list(coordinator.logs))
    time = np.array(list(coordinator.times))
    lambdas = np.array(list(coordinator.lambdas))
    solutions = np.array(list(coordinator.solutions))

    np.save(path + header + 'logs.npy', np.hstack([time.reshape(-1, 1), logs]).astype(np.float32))
    np.save(path + header + 'lambdas.npy', np.hstack([time.reshape(-1, 1), lambdas]).astype(np.float32))
    np.save(path + header + 'solutions.npy', np.hstack([time.reshape(-1, 1), solutions]).astype(np.float32))

def experiment(dat_path = 'd201600.dat', output_path='./', exp_name = 'DALSR',time = 100, sub_nums=20, M = 70, r = 0.05):
    para = load_dat(dat_path)
    
    sub_paras = []
    for j in range(sub_nums):
        sub_para = {
            'objective_g': para['objective'][j],
            'objective_c': [1 for i in range(len(para['objective'][j]))],
            'constraints': para['constraints'][j],
            'bs': para['bs'][j],
            'ub': 1,
            'lb': 0
        }
        sub_paras.append(sub_para)
        
        
    coordinator = Coordinator(time=time, M=M, r=r)
    subproblems = [Solver(i) for i in range(sub_nums)]
    for index, subproblem in enumerate(subproblems):
        subproblem.init(sub_paras[index], coordinator)
    coordinator.init(para, subproblems)
    coordinator.run()
    
    save_result(coordinator, path=output_path, header=exp_name)

experiment(
    dat_path = 'd201600.dat', 
    output_path = './',
    exp_name = 'SLR',
    time = 600,
    sub_nums = 20,
    M = 75,
    r = 0.05   
)

