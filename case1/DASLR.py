#!/usr/bin/env python
# coding: utf-8

import numpy as np
import copy
import time
import docplex.mp.model as cpx
import queue
from multiprocessing import Process, Queue, Manager
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm


class Coordinator(Process):
    '''
    Coordinator for updating and broadcasting multipliers.
    '''
    def __init__(self, M = 50, r = 0.05, L_star = 15.6, nums = 10, threshold = 0.1):
        super().__init__()
        self._M = M
        self._r = r
        self._L_star = L_star
        self.nums = nums
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
        
        self._lambda_queues = None
        ## Elements format in x_queue (index, x)
        self._x_queue = Queue()
        
        self.logs = None
        
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
        
        self._k, self._lambda = self._init_lambda()
        self._lambda_queues = [sub._lambda_queue for sub in self._subproblems]
        
        self.logs = Manager().list()
        
        ## init lambda^0 and send to subproblems
        for queue in self._lambda_queues:
            queue.put((self._k, self._lambda))
        
        ## listen and wait for x^0
        while self._x_queue.qsize() != len(self._subproblems):
            continue
        else:
            ## Get x^0
            while self._x_queue.qsize() != 0:
                index, x_0 = self._x_queue.get()
                self._solutions[index] = x_0
                
        ## Reset the paprameters
        self._k = 0
                
        ## Update subgradient with x^0
        self._g = self._get_g()
        ## Update surrogate_dual with x^0 and g^0
        self._surrogate_dual = self._get_surrogate_dual()
        ## Update s^0 with x^0, surrogate_dual and g^0
        ## self._s = (self._L_star - self._surrogate_dual) / np.linalg.norm(self._g)
        self._s = 0.019
        
        ## Resend lambda to all solvers
        for queue in self._lambda_queues:
            queue.put((self._k, self._lambda))

        
    def run(self):
        '''
        Start and keep updating multipliers.
        '''
        start_time = time.time()
        ## Ignoring the stop creteria for debug
        self._dual_gap = 99999
        ## Start from k = 1
        while self._k <= self.nums and self._dual_gap >= self.threshold:
            ## Update lambda
            index, x = self._x_queue.get()
            self._solutions[index] = x
            self._old_g = self._g
            self._g = self._get_g()
            self._k = self._k + 1
            self._s = self._get_s()
            self._lambda = self._lambda + self._s * self._g
            self._project_lambda()
            
            current_time_diff = time.time() - start_time
            self.logs.append([current_time_diff, self._lambda])
            if current_time_diff > 100:
                break
                
            ## Send lambda to all solvers
            for queue in self._lambda_queues:
                queue.put_nowait((self._k, self._lambda))
        for queue in self._lambda_queues:
            queue.put_nowait('stop')
        end_time = time.time() - start_time
        
    def _init_lambda(self):
        return (-1, np.zeros(shape = len(self.constraints)))
    
    def _project_lambda(self):
        for i,x in enumerate(self._lambda):
            if x < 0:
                self._lambda[i] = 0
    
    def _get_g(self):
        '''
        This method can only use after update x
        '''
        return np.array(
            [np.sum([self.constraints[i][x] * self._solutions[x] for x in range(len(self._subproblems))]) + self.bs[i] 
         for i in range(len(self._lambda))]
        )
    
    def _get_s(self):
        if self._k == 0:
            return self._s
        p = 1 - (1 / (self._k ** self._r))
        alpha = 1 - (1 / (self._M * (self._k ** p)))
        s_new = alpha * self._s * (np.linalg.norm(self._old_g, ord=2) / np.linalg.norm(self._g, ord=2))
        return s_new
    
    def _get_surrogate_dual(self):
        '''
        This method can only use after update x and g!
        '''
        return np.sum([self._solutions[x] * self.objective[x] for x in range(len(self._subproblems))]) + np.sum([self._lambda[i] * self._g[i] for i in range(len(self._lambda))])
    


class Solver(Process):
    '''
    Solver for solving subproblems and sending them back to coordinator.
    
    index: int value indicates the number of subproblem.
    '''
    def __init__(self, index):
        super().__init__()
        self._lambda = None
        self._x = None
        self._k = 0
        
        self._x_queue = None
        self._lambda_queue = Queue()
        self._lambda_stack = queue.LifoQueue()
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
        self._x_queue = coordinator._x_queue
        
        self._k = -1
        self.objective_g = para['objective_g']
        self.objective_c = para['objective_c']
        self.constraints = para['constraints']
        self.bs = para['bs'] 
        self.ub = para['ub']
        self.lb = para['lb']
        
    def run(self):        
        while True:
            while not self._lambda_queue.empty():
                self._lambda_stack.put(self._lambda_queue.get())
            if self._lambda_stack.qsize() == 0:
                continue
            else:
                message = self._lambda_stack.get()
                
            if message == 'stop':
                break
            if message[0] > self._lambda_index:
                self._lambda = message[1]
                self._lambda_index = message[0]
                
                ## Solve solution with latest lambda
                self.solve()
                self._k = self._k + 1
                
                ## simulating the time delay of 5g transmission
                time.sleep(0.001)
                self._x_queue.put_nowait((self._index, self._x))
                
    def solve(self,):
        model = cpx.Model(name="MIP Model")
        model.context.cplex_parameters.threads = 1
        x_var = model.continuous_var(lb = self.lb, ub = self.ub, name = self.name)
        ## pass | set constraints latter
        objective = model.sum([self._lambda[i] * self.objective_c[i] * x_var for i in range(len(self._lambda))]) + self.objective_g * x_var
        model.minimize(objective)
        result = model.solve()
        self._x = result.get_all_values()[0]
    
    def __str__(self):
        return self._name
    

def fix_timestamp(logs):
    fix = list(logs)

    fix_x = [x[0] // 0.1 for x in fix]
    fix_y = [x[1] for x in fix]

    fix = dict()

    for i in range(len(fix_x)):
        fix[fix_x[i]] = fix_y[i]

    logs = []
    for key in fix:
        logs.append([key / 10, fix[key]])
    return logs


para = {
    'objective': [1,2,3,1,2,3],
    'constraints': [[-1,-3,-5,-1,-3,-5], [-2,-1.5,-5,-2,-0.5,-1]],
    'bs': [26, 16]
}


sub_paras = []
for i in range(6):
    sub_para = {
        'objective_g': para['objective'][i],
        'objective_c': [para['constraints'][0][i], para['constraints'][1][i]],
        'constraints': [],
        'bs': [],
        'ub': 3,
        'lb': 0
    }
    sub_paras.append(sub_para)


logs_all = []
for i in tqdm(range(100)):
    sequence = np.random.randint(0, 6, 1000000)
    coordinator = Coordinator(nums=99999999)

    subproblems = [Solver(i) for i in range(6)]
    for index, subproblem in enumerate(subproblems):
        subproblem.init(sub_paras[index], coordinator)
        subproblem.start()
    coordinator.init(para, subproblems)
    coordinator.start()
    coordinator.join()
    logs = fix_timestamp(coordinator.logs)
    logs_all.append(logs)


f = open('DASLR_100.txt','w')
f.write(str(logs_all))
f.close()

