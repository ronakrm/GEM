import torch
import numpy as np
import ot
from torch.autograd import grad as agrad
import time

from emd_torch import torch_greedy_primal_dual

from diff_test import takeStep

# def demdTorch(inp):
#     x = inp.detach().cpu().numpy()
#     obj = demd_func(x)
#     return torch.Tensor([obj]).requires_grad_(True).to(inp.device)

def torch_demd_func(x):
    log = torch_greedy_primal_dual(x)
    return log['primal objective']

# class dEMD(torch.autograd.Function):

#     @staticmethod
#     def forward(ctx, inp):
#         out = demdTorch(inp)
#         ctx.save_for_backward(inp, out)
#         return out
    
#     @staticmethod
#     def backward(ctx, grad_output):

#     #     # assert grad_output.shape == self.output.shape
#     #     # grad_output_numpy = grad_output#.detach().cpu().numpy()

#         inp, out = ctx.saved_tensors

#     #     # finite diff step
#         inp_prime = inp + epsilon * grad_output

#         out_prime = demdTorch(inp_prime)
#         gradient = -(out - out_prime) / epsilon
#         return grad_output

# class dEMD(torch.nn.Module):
#     def __init__(self):
#         super(self.__name__, self).__init__()

#         self.fc1 = torch.nn.Linear()

#     def forward(x):
#         return torch_demd_func(x)


def minimize(func, x_0, niters=100, lr=0.1, verbose=False):

    # opt = torch.optim.SGD(x, lr=lr)

    x = x_0

    if verbose:
        tic = time.time()
        funcval = func(x)
        print(time.time()-tic, ' seconds for forward.')

        tic = time.time()
        grad, = agrad(outputs=funcval, inputs=x, allow_unused=True)
        print(time.time()-tic, ' seconds for gradient.')
    else:
        funcval = func(x)
        grad, = agrad(outputs=funcval, inputs=x, allow_unused=True)

    #print(grad)
    #print(torch_greedy_primal_dual(x)['dual'])
    #import pdb; pdb.set_trace()

    gn = torch.linalg.norm(grad)

    print(f'Inital:\t\tObj:\t{funcval:.4f}\tGradNorm:\t{gn:.4f}')

    for i in range(niters):


        funcval = func(x)

        # opt.zero_grad()

        x = takeStep(x, grad, lr)

        x = x / (torch.sum(x, 1).unsqueeze(-1))

        gn = np.linalg.norm(grad)
        funcval = func(x)
        grad, = agrad(outputs=funcval, inputs=x, allow_unused=True)
        
        if i % 100 == 0:
            print(f'Iter {i:2.0f}:\tObj:\t{funcval:.4f}\tGradNorm:\t{gn:.4f}')

    print(x)
    return



if __name__ == "__main__":
    
    np.random.seed(0)

    print('*'*10)
    print('*** 2 Fixed Dists with 6 Bins ***')
    #######
    n = 5  # nb bins
    x = np.arange(n, dtype=np.float64).reshape((n, 1))
    M = ot.utils.dist(x, metric='minkowski')

    a1 = np.array([0.5, 0.2, 0.1, 0.1, 0.1])
    a2 = np.array([0.2, 0.1, 0.2, 0.3, 0.2])
    a3 = np.array([0.1, 0.1, 0.5, 0.2, 0.1])
    data = [a1, a2, a3]
    d = len(data)
    print(data)

    # data = np.array(data)
    # data = vectorize(data, vecsize)

    ta1 = torch.Tensor(a1)
    ta2 = torch.Tensor(a2)
    ta3 = torch.Tensor(a3)
    torch_data = [ta1, ta2, ta3]
    torch_data = torch.stack(torch_data).clone().requires_grad_(requires_grad=True)

    # data = data.clone().requires_grad_(requires_grad=True)

    # model = dEMD(epsilon=epsilon)
    func = torch_demd_func
    # func = dEMD(torch_data)
    minimize(func, torch_data, niters=500, lr=0.001, verbose=True)

   
