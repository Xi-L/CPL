import torch
import numpy as np

# device
device = 'cpu'

def get_problem(name, *args, **kwargs):
    name = name.lower()
    
    PROBLEM = {
        'ackley': ackley,
        'rosenbrock': rosenbrock,
        'himmelblau': himmelblau,
        'regression_f1': regression_f1,
        'regression_f2': regression_f2,
        'regression_f3': regression_f3,
        'regression_f4': regression_f4,
 }

    if name not in PROBLEM:
        raise Exception("Problem not found.")
    
    return PROBLEM[name](*args, **kwargs)

class ackley():
    def __init__(self, n_dim = 2):
        self.n_dim = n_dim
        
    def f_func(self, x_input):
        x = x_input[:,0]
        y = x_input[:,1]
        
        f = -20 * torch.exp(-0.2 * torch.sqrt(0.5 * (x**2 + y**2))) - torch.exp(0.5 * (torch.cos(2 * np.pi * x) + torch.cos(2 * np.pi * y))) + np.e + 20
        
        return f
        
    def evaluate(self, x_input, t):
       
        t = 1 - t + 0.0001
        
        u = torch.randn([t.shape[0],2]).to(device)
        f = self.f_func(x_input)
        grad_hat = 1 / t * (self.f_func(x_input + t.expand(t.shape[0],2) * u) - f).repeat(2,1).T * u

        return f, grad_hat


class rosenbrock():
    def __init__(self, n_dim = 2):
        self.n_dim = n_dim
        
        
    def evaluate(self, x_input, t):
        x = x_input[:,0]
        y = x_input[:,1]
        t = 1.5 * (1 - t.T[0])
        
        # f = 100 * (y - x **2) ** 2 + (1 - x) ** 2
        F = 100 * x ** 4 + (-200 * y + 600 * t ** 2 + 1) * x**2 - 2 * x + 100 * y**2 - 200 * t**2 *y + (300 * t**4 + 101 * t**2 + 1)
        dFdx = 400 * x**3 + 2 * (-200 * y + 600 * t ** 2 + 1) * x - 2
        dFdy = - 200 * x**2 + 200 * y - 200 * t**2
        
        return F, torch.stack([dFdx, dFdy]).T

class himmelblau():
    def __init__(self, n_dim = 2):
        self.n_dim = n_dim
        
        
    def evaluate(self, x_input, t):
        x = x_input[:,0]
        y = x_input[:,1]
        t = 2 * (1 - t.T[0])
        
        # f = (x**2 + y - 11)**2 + (x + y**2 - 7)**2
        F = x**4 + (2*y + 6*t**2 - 21) * x**2 + (2 * y**2 + 2 * t**2 - 14) * x + y**4 + (6 * t**2 - 13) * y**2 + (2 * t**2 - 22) * y +(6 * t**4 - 34 * t**2 + 170)
        dFdx = 4 * x**3 + 2* (2*y + 6*t**2 - 21) * x + (2 * y**2 + 2 * t**2 - 14)
        dFdy = 2 * x**2 + 4 * x * y + 4 * y**3 + 2 * (6 * t**2 - 13) * y +  (2 * t**2 - 22) 
        
        return F, torch.stack([dFdx, dFdy]).T


class regression_f1():
    def __init__(self, n_dim = 3):
        self.n_dim = n_dim
        
    def evaluate(self, beta, t):
        
        x = torch.rand(100, generator = torch.random.manual_seed(1)) * 10 - 5
        eps = torch.rand(100, generator = torch.random.manual_seed(1)) * 0.1
        
        y = 0.5 * torch.sin(x) + 0.3 * torch.cos(2 * x) + 2 * torch.cos(3 * x)  + eps
        x = x.float().to(device)
        y = y.float().to(device)
        
        X = torch.stack([torch.sin(x), torch.cos(2*x), torch.cos(3 * x)]).float()
        XX = torch.matmul(X, X.T).float()
        
        #F = t * torch.sum((torch.matmul(X.T,beta.T).T - y) ** 2) + (1 - t) * torch.sum(beta**2)
        F1 = torch.sum((torch.matmul(X.T,beta.T).T - y) ** 2)
        F2 = torch.sum(beta**2)
        F_grad = t * (2 * torch.matmul(XX, beta.T).T - 2 * torch.matmul(X, y)) + (1 - t) * 2  * beta
       
        return torch.stack([F1.data, F2.data]), F_grad
    
    def evaluate_test(self, beta, t):
        
        x = torch.rand(100, generator = torch.random.manual_seed(123)) * 10 - 5
        y = 0.5 * torch.sin(x) + 0.3 * torch.cos(2 * x) + 2 * torch.cos(3 * x)  
        x = x.float().to(device)
        y = y.float().to(device)
        
        X = torch.stack([torch.sin(x), torch.cos(2*x), torch.cos(3 * x)]).float()
        XX = torch.matmul(X, X.T).float()
        
        #F = t * torch.sum((torch.matmul(X.T,beta.T).T - y) ** 2) + (1 - t) * torch.sum(beta**2)
        F1 = torch.sum((torch.matmul(X.T,beta.T).T - y) ** 2)
        F2 = torch.sum(beta**2)
        F_grad = t * (2 * torch.matmul(XX, beta.T).T - 2 * torch.matmul(X, y)) + (1 - t) * 2  * beta
       
        return torch.stack([F1.data, F2.data]), F_grad

class regression_f2():
    def __init__(self, n_dim = 3):
        self.n_dim = n_dim
        
        
    def evaluate(self, beta, t):
        
        x = torch.rand(100, generator = torch.random.manual_seed(2)) * 10 - 5
        eps = torch.rand(100, generator = torch.random.manual_seed(2)) * 0.1
    
        y = 1 * torch.cos(x) + 0.2 * torch.sin(2 * x) + 0.5 * torch.sin(3 * x)  + eps
        x = x.float().to(device)
        y = y.float().to(device)
        
        X = torch.stack([torch.cos(x), torch.sin(2*x), torch.sin(3 * x)]).float()
        XX = torch.matmul(X, X.T).float()
        
        #F = t * torch.sum((torch.matmul(X.T,beta.T).T - y) ** 2) + (1 - t) * torch.sum(beta**2)
        F1 = torch.sum((torch.matmul(X.T,beta.T).T - y) ** 2)
        F2 = torch.sum(beta**2)
        F_grad = t * (2 * torch.matmul(XX, beta.T).T - 2 * torch.matmul(X, y)) + (1 - t) * 2  * beta
       
        return torch.stack([F1.data, F2.data]), F_grad
    
    def evaluate_test(self, beta, t):
        
        x = torch.rand(100, generator = torch.random.manual_seed(234)) * 10 - 5
        y = 1 * torch.cos(x) + 0.2 * torch.sin(2 * x) + 0.5 * torch.sin(3 * x)  #+ eps
        x = x.float().to(device)
        y = y.float().to(device)
        
        
        X = torch.stack([torch.cos(x), torch.sin(2*x), torch.sin(3 * x)]).float()
        XX = torch.matmul(X, X.T).float()
        
        #F = t * torch.sum((torch.matmul(X.T,beta.T).T - y) ** 2) + (1 - t) * torch.sum(beta**2)
        F1 = torch.sum((torch.matmul(X.T,beta.T).T - y) ** 2)
        F2 = torch.sum(beta**2)
        F_grad = t * (2 * torch.matmul(XX, beta.T).T - 2 * torch.matmul(X, y)) + (1 - t) * 2  * beta
       
        return torch.stack([F1.data, F2.data]), F_grad
    
class regression_f3():
    def __init__(self, n_dim = 3):
        self.n_dim = n_dim
        
        
    def evaluate(self, beta, t):
        
        x = torch.rand(100, generator = torch.random.manual_seed(4)) * 10 - 5
        eps = torch.rand(100, generator = torch.random.manual_seed(4)) * 0.1
        y  = 1 * torch.exp(0.25 * x) - 0.2 * torch.cos(x) + 0.5 * torch.sin(4 * x)  + eps
        x = x.float().to(device)
        y = y.float().to(device)
        
        X = torch.stack([torch.exp(0.25 * x), torch.cos(x), torch.sin(4 * x)]).float()
        XX = torch.matmul(X, X.T).float()
        
        #F = t * torch.sum((torch.matmul(X.T,beta.T).T - y) ** 2) + (1 - t) * torch.sum(beta**2)
        F1 = torch.sum((torch.matmul(X.T,beta.T).T - y) ** 2)
        F2 = torch.sum(beta**2)
        F_grad = t * (2 * torch.matmul(XX, beta.T).T - 2 * torch.matmul(X, y)) + (1 - t) * 2  * beta
       
        return torch.stack([F1.data, F2.data]), F_grad
    
    def evaluate_test(self, beta, t):
        
        x = torch.rand(100, generator = torch.random.manual_seed(456)) * 10 - 5
       
        y  = 1 * torch.exp(0.25 * x) - 0.2 * torch.cos(x) + 0.5 * torch.sin(4 * x)  #+ eps
        x = x.float().to(device)
        y = y.float().to(device)
        
        
        X = torch.stack([torch.exp(0.25 * x), torch.cos(x), torch.sin(4 * x)]).float()
        XX = torch.matmul(X, X.T).float()
        
        #F = t * torch.sum((torch.matmul(X.T,beta.T).T - y) ** 2) + (1 - t) * torch.sum(beta**2)
        F1 = torch.sum((torch.matmul(X.T,beta.T).T - y) ** 2)
        F2 = torch.sum(beta**2)
        F_grad = t * (2 * torch.matmul(XX, beta.T).T - 2 * torch.matmul(X, y)) + (1 - t) * 2  * beta
       
        return torch.stack([F1.data, F2.data]), F_grad
    
class regression_f4():
    def __init__(self, n_dim = 3):
        self.n_dim = n_dim
        
        
    def evaluate(self, beta, t):
        
        x = torch.rand(100, generator = torch.random.manual_seed(5)) * 10 - 5
        eps = torch.rand(100, generator = torch.random.manual_seed(5)) * 0.2
        y  = 2 * torch.log(0.25 * torch.abs(x)) + 3 * torch.sin(6 * x) + 4 * torch.cos(0.5 * x)  + eps
        x = x.float().to(device)
        y = y.float().to(device)
        
        X = torch.stack([torch.log(0.25 * torch.abs(x)), torch.sin(6 * x), torch.cos(0.5 * x)]).float()
        XX = torch.matmul(X, X.T).float()
        
        #F = t * torch.sum((torch.matmul(X.T,beta.T).T - y) ** 2) + (1 - t) * torch.sum(beta**2)
        F1 = torch.sum((torch.matmul(X.T,beta.T).T - y) ** 2)
        F2 = torch.sum(beta**2)
        F_grad = t * (2 * torch.matmul(XX, beta.T).T - 2 * torch.matmul(X, y)) + (1 - t) * 2  * beta
       
        return torch.stack([F1.data, F2.data]), F_grad
    
    def evaluate_test(self, beta, t):
        
        x = torch.rand(100, generator = torch.random.manual_seed(567)) * 10 - 5
      
        y  = 2 * torch.log(0.25 * torch.abs(x)) + 3 * torch.sin(6 * x) + 4 * torch.cos(0.5 * x)   #+ eps
        x = x.float().to(device)
        y = y.float().to(device)
        
        X = torch.stack([torch.log(0.25 * torch.abs(x)), torch.sin(6 * x), torch.cos(0.5 * x)]).float()
        XX = torch.matmul(X, X.T).float()
        
        #F = t * torch.sum((torch.matmul(X.T,beta.T).T - y) ** 2) + (1 - t) * torch.sum(beta**2)
        F1 = torch.sum((torch.matmul(X.T,beta.T).T - y) ** 2)
        F2 = torch.sum(beta**2)
        F_grad = t * (2 * torch.matmul(XX, beta.T).T - 2 * torch.matmul(X, y)) + (1 - t) * 2  * beta
       
        return torch.stack([F1.data, F2.data]), F_grad

