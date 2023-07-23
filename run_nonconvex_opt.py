"""
Continuation Path Learning (CPL) for Non-convex Optimization
"""
import torch

from problem import get_problem
from model import ParetoSetModel

import timeit

# device
device = 'cpu'

# number of independent runs
n_run = 20 

# instance name, from ['ackley', 'himmelblau', 'rosenbrock', ]
test_ins = 'ackley'


if test_ins == 'ackley':

    # number of learning steps
    n_steps = 225 
    # number of sampled homotopy levels at each step
    n_levels = 4 
    # number of local search
    n_local_search = 100 
    # initial solution
    init_point = torch.tensor([5,5]).to(device)
    
    
if test_ins == 'himmelblau':

    # number of learning steps
    n_steps = 450
    # number of sampled homotopy levels at each step
    n_levels = 4
    # number of local search
    n_local_search = 200
    # initial solution
    init_point = torch.tensor([-3,-2]).to(device) 

    
if test_ins == 'rosenbrock':

    # number of learning steps
    n_steps = 4500 
    # number of sampled homotopy levels at each step
    n_levels = 4 
    # number of local search
    n_local_search = 2000
    # initial solution
    init_point = torch.tensor([-3,2]).to(device) 
    


problem = get_problem(test_ins)
n_dim = problem.n_dim
    
# repeatedly run the algorithm n_run times
value_list = []
for run_iter in range(n_run):
    
    # model initialization
    psmodel = ParetoSetModel(n_dim)
    psmodel.to(device)
    psmodel.train()
    
    
    # let the model generate the same initial solutions with other homotopy methods
    optimizer = torch.optim.Adam(psmodel.parameters(), lr=1e-4)
    
    for i in range(1000):
    
        t = torch.ones([1,1]).to(device)
        x = psmodel(t)
        loss = torch.abs(x - init_point).sum()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    # path model training
    start = timeit.default_timer()
    optimizer = torch.optim.Adam(psmodel.parameters(), lr=5e-3)
    
    for t_step in range(n_steps):
        
        # sample n_levels homotopy levels
        t = torch.rand([n_levels,1]).to(device)
        t[0] = 1
      
        # get the current coressponding solutions and gradients
        x = psmodel(t)
        value, grad = problem.evaluate(x,t)

        # gradient-based continuation path model update 
        optimizer.zero_grad()
        psmodel(t).backward(grad)
        optimizer.step()  
        
    # optional local search for homotopy level t = 1
    t = torch.ones([1,1]).to(device)
    for t_step in range(n_local_search):
        
        x = psmodel(t)
        value, grad = problem.evaluate(x,t)
       
        optimizer.zero_grad()
        psmodel(t).backward(grad)
        optimizer.step()  
        
    
    # pring the final solution with homotopy level t = 1
    t = torch.ones([1,1]).to(device)
    x = psmodel(t)
    value, grad = problem.evaluate(x,t)
    print('Run', run_iter+1)
    print('Solution:', x)
    print('Value:', value.item())
    value_list.append(value)
    
    stop = timeit.default_timer()
        
    print('Time: ', stop - start)  
    print("************************************************************")

avg_value = torch.mean(torch.tensor(value_list))
print('Average Value: ', avg_value.item())
    