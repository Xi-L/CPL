"""
Continuation Path Learning (PSL) for Noisy Regression
"""
import torch

from problem import get_problem
from model import ParetoSetModel

import numpy as np
import matplotlib.pyplot as plt

# device
device = 'cpu'

test_ins = 'regression_f1'

# number of learning steps
n_steps = 10000 

# number of sampled homotopy levels at each step
n_levels = 5 

problem = get_problem(test_ins)
n_dim = problem.n_dim
    
# model initialization
psmodel = ParetoSetModel(n_dim)
psmodel.to(device)
psmodel.train()
    
# optimizer
optimizer = torch.optim.Adam(psmodel.parameters(), lr=1e-4)

for t_step in range(n_steps):
    
    # sample n_levels homotopy levels
    t = torch.rand([n_levels,1]).to(device)
    t[0] = 0
    # get the current coressponding solutions and gradients
    x = psmodel(t).float()
    value, grad = problem.evaluate(x,t)
   
    # gradient-based pareto set model update 
    optimizer.zero_grad()
    psmodel(t).backward(grad)
    optimizer.step()  
    
    if t_step%500 == 0:
        print("Step", t_step)
    
print("************************************************************")
    
value_list = []
for k in range(101):
    t = torch.ones([1,1]).to(device) * k * 0.01
    x = psmodel(t).float()
    value, grad = problem.evaluate_test(x,t)
    value_list.append(value)
    
value_mat = torch.stack(value_list)
value_mat_np = np.array(value_mat.cpu())
    
fig = plt.figure()

t_vec = np.arange(101) * 0.01

plt.plot(t_vec, value_mat_np[:,0], color= 'C1', label= 'Validation Loss', lw = 2)
plt.yscale('log')

plt.xlabel('t', fontsize = 20)
plt.ylabel('Prediction Loss', fontsize = 20)
plt.grid()


